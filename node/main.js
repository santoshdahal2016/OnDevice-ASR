#!/usr/bin/env node

/**
 * Node.js ONNX RNNT inference script
 * This script uses ONNX models for encoder/decoder/joiner inference
 */

const fs = require('fs-extra');
const path = require('path');
const yargs = require('yargs');
const ort = require('onnxruntime-node');
const wav = require('node-wav');
const NeMoStyleAudioPreprocessor = require('./nemo_style_preprocessing');

// JS-only pipeline

/**
 * Load metadata from ONNX model
 * @param {string} modelPath - Path to ONNX model
 * @returns {Object} metadata - Model metadata
 */
async function loadONNXMetadata(modelPath) {
    try {
        // For now, we'll use default values since reading ONNX metadata in Node.js requires additional libraries
        // In a full implementation, you'd use a library like 'onnx-js' or parse the protobuf directly
        return {
            sample_rate: '16000',
            chunk_shift: '416',
            vocab_size: '1024',
            pred_rnn_layers: '2',
            pred_hidden: '640',  // Model expects 640
            window_size: '0.025',
            window_stride: '0.01',
            cache_last_channel_dim1: '18',
            cache_last_channel_dim2: '8',
            cache_last_channel_dim3: '640',  // Model expects 640
            cache_last_time_dim1: '18',
            cache_last_time_dim2: '640',     // Model expects 640
            cache_last_time_dim3: '3'
        };
    } catch (error) {
        console.error('Error loading ONNX metadata:', error);
        return {};
    }
}

/**
 * Load vocabulary from tokens file
 * @param {string} tokensFile - Path to tokens file
 * @returns {Array<string>} vocabulary - Array of tokens
 */
async function loadVocabulary(tokensFile) {
    try {
        const content = await fs.readFile(tokensFile, 'utf-8');
        const vocabulary = [];
        
        const lines = content.trim().split('\n');
        for (const line of lines) {
            const parts = line.trim().split(/\s+/);
            if (parts.length >= 2) {
                // Join all but last part (which is the index)
                const token = parts.slice(0, -1).join(' ');
                vocabulary.push(token);
            }
        }
        
        return vocabulary;
    } catch (error) {
        console.error('Error loading vocabulary:', error);
        return [];
    }
}

/**
 * Advanced audio preprocessing using NeMo-style mel-spectrogram extraction
 * @param {string} audioPath - Path to audio file
 * @param {number} sampleRate - Target sample rate
 * @returns {Object} - Processed audio data
 */
async function preprocessAudio(audioPath, sampleRate = 16000) {
    try {
        // Load audio file
        const buffer = await fs.readFile(audioPath);
        const audioData = wav.decode(buffer);
        
        // Convert to Float32Array and resample if needed
        let audioSignal = new Float32Array(audioData.channelData[0]);
        
        if (audioData.sampleRate !== sampleRate) {
            console.log(`Resampling from ${audioData.sampleRate}Hz to ${sampleRate}Hz`);
            // Simple resampling (for production, use a proper resampling library)
            const ratio = audioData.sampleRate / sampleRate;
            const newLength = Math.ceil(audioSignal.length / ratio);
            const resampled = new Float32Array(newLength);
            
            for (let i = 0; i < newLength; i++) {
                const srcIndex = i * ratio;
                const leftIndex = Math.floor(srcIndex);
                const rightIndex = Math.min(leftIndex + 1, audioSignal.length - 1);
                const fraction = srcIndex - leftIndex;
                
                // Linear interpolation (closer to librosa's resampling)
                resampled[i] = audioSignal[leftIndex] * (1 - fraction) + audioSignal[rightIndex] * fraction;
            }
            audioSignal = resampled;
        }

        // Use TorchScript preprocessor unless JS is forced
        // JS NeMo-style preprocessor path (always)
        {
            const preprocessor = new NeMoStyleAudioPreprocessor({
                sampleRate: sampleRate,
                nWindowSize: Math.floor(0.025 * sampleRate),
                nWindowStride: Math.floor(0.01 * sampleRate),
                window: 'hann',
                normalize: 'per_feature',
                preemph: 0.97,
                nfilt: 80,
                lowfreq: 0,
                highfreq: sampleRate / 2,
                log: true,
                logZeroGuardType: 'add',
                logZeroGuardValue: Math.pow(2, -24),
                dither: 1e-5,
                magPower: 2.0,
                exactPad: false,
                padTo: 16,
                padValue: 0.0,
                frameSplicing: 1
            });
            console.log(`Processing ${audioSignal.length} samples at ${sampleRate}Hz`);
            const result = preprocessor.forward(audioSignal);
            const numFeatures = result.melSpectrogram[0].length;
            const numFrames = result.melSpectrogram.length;
            const processedSignal = new Float32Array(1 * numFeatures * numFrames);
            for (let t = 0; t < numFrames; t++) {
                for (let f = 0; f < numFeatures; f++) {
                    processedSignal[f * numFrames + t] = result.melSpectrogram[t][f];
                }
            }
            const processedLength = new BigInt64Array([BigInt(result.seqLen)]);
            const shape = [1, numFeatures, numFrames];
            console.log(`Preprocessed to shape: [${shape.join(', ')}]`);
            return { processedSignal, processedLength, shape };
        }
        
    } catch (error) {
        console.error('Error preprocessing audio:', error);
        throw error;
    }
}

/**
 * Hypothesis class for RNNT decoding - JavaScript equivalent of NEMO's Hypothesis
 */
class Hypothesis {
    constructor() {
        this.y_sequence = [];        // Predicted token sequence
        this.score = 0.0;            // Log probability score
        this.timestep = [];          // Timestep for each token
        this.last_token = null;      // Last predicted token
        this.dec_state = null;       // Decoder state
        this.text = "";              // Decoded text
        this.confidence = [];        // Per-token confidence scores
        this.alignments = [];        // Token alignments
        this.frame_confidence = [];  // Frame-level confidence
    }
    
    /**
     * Add a new token to the hypothesis
     */
    addToken(token, score, timestep, confidence = null) {
        this.y_sequence.push(token);
        this.score += score;
        this.timestep.push(timestep);
        this.last_token = token;
        
        if (confidence !== null) {
            this.confidence.push(confidence);
        }
    }
    
    /**
     * Clone the hypothesis
     */
    clone() {
        const cloned = new Hypothesis();
        cloned.y_sequence = [...this.y_sequence];
        cloned.score = this.score;
        cloned.timestep = [...this.timestep];
        cloned.last_token = this.last_token;
        cloned.dec_state = this.dec_state; // Shallow copy for now
        cloned.text = this.text;
        cloned.confidence = [...this.confidence];
        cloned.alignments = [...this.alignments];
        cloned.frame_confidence = [...this.frame_confidence];
        return cloned;
    }
}

/**
 * ONNX RNNT Inference class with NEMO-style result matching
 */
class ONNXRNNTInference {
    constructor(modelDir, tokensFile) {
        this.modelDir = modelDir;
        this.tokensFile = tokensFile;
        this.vocabulary = [];
        this.metadata = {};
        
        // Model sessions
        this.encoderSession = null;
        this.decoderSession = null;
        this.joinerSession = null;
        
        // Model parameters
        this.sampleRate = 16000;
        this.chunkShift = 416;
        this.vocabSize = 1024;
        this.predRnnLayers = 2;
        this.predHidden = 640;  // Model expects 640 dimensions
        this.windowSize = 425;
        
        // Cache dimensions (updated to match model expectations)
        this.cacheLastChannelDim1 = 18;
        this.cacheLastChannelDim2 = 8;
        this.cacheLastChannelDim3 = 640;  // Model expects 640
        this.cacheLastTimeDim1 = 18;
        this.cacheLastTimeDim2 = 640;     // Model expects 640
        this.cacheLastTimeDim3 = 3;
        
        // Decoding parameters
        this.maxSymbolsPerStep = 30;
        this.preserveAlignments = true;
        this.preserveFrameConfidence = true;
        this.confidenceThreshold = 0.5;
    }
    
    /**
     * Initialize the inference engine
     */
    async initialize() {
        try {
            // Load metadata
            this.metadata = await loadONNXMetadata(path.join(this.modelDir, 'encoder.onnx'));
            
            // Parse metadata
            this.sampleRate = parseInt(this.metadata.sample_rate || '16000');
            this.chunkShift = parseInt(this.metadata.chunk_shift || '416');
            this.vocabSize = parseInt(this.metadata.vocab_size || '1024');
            this.predRnnLayers = parseInt(this.metadata.pred_rnn_layers || '2');
            this.predHidden = parseInt(this.metadata.pred_hidden || '640');
            this.windowSize = this.chunkShift + 9; // Approximate pre_encode_cache_size
            
            // Load vocabulary
            this.vocabulary = await loadVocabulary(this.tokensFile);
            if (this.vocabulary.length > 0) {
                this.vocabSize = this.vocabulary.length;
            }
            
            // Load ONNX models
            const providers = ['cpu']; // Start with CPU, can add CUDA later
            
            this.encoderSession = await ort.InferenceSession.create(
                path.join(this.modelDir, 'encoder.onnx'),
                { providers }
            );
            
            this.decoderSession = await ort.InferenceSession.create(
                path.join(this.modelDir, 'decoder.onnx'),
                { providers }
            );
            
            this.joinerSession = await ort.InferenceSession.create(
                path.join(this.modelDir, 'joiner.onnx'),
                { providers }
            );
            
            console.log('Loaded ONNX RNNT model with:');
            console.log(`  Sample rate: ${this.sampleRate}`);
            console.log(`  Streaming window size: ${this.windowSize}`);
            console.log(`  Chunk shift: ${this.chunkShift}`);
            console.log(`  Vocabulary size: ${this.vocabSize}`);
            console.log(`  Vocabulary preview: ${this.vocabulary.slice(0, 10).join(', ')}...`);
            
        } catch (error) {
            console.error('Error initializing ONNX RNNT inference:', error);
            throw error;
        }
    }
    
    /**
     * Run encoder
     * @param {Float32Array} processedSignal - Preprocessed audio signal
     * @param {BigInt64Array} processedLength - Length of processed signal
     * @param {Array<number>} shape - Shape of the input tensor [batch, features, time]
     * @returns {Object} - Encoder outputs
     */
    async runEncoder(processedSignal, processedLength, shape) {
        try {
            const inputTensor = new ort.Tensor('float32', processedSignal, shape);
            const lengthTensor = new ort.Tensor('int64', processedLength, [1]);
            
            const feeds = {};
            feeds[this.encoderSession.inputNames[0]] = inputTensor;
            feeds[this.encoderSession.inputNames[1]] = lengthTensor;
            
            const results = await this.encoderSession.run(feeds);
            
            return {
                encoded: results[this.encoderSession.outputNames[0]],
                encodedLengths: results[this.encoderSession.outputNames[1]]
            };
        } catch (error) {
            console.error('Error running encoder:', error);
            throw error;
        }
    }
    
    /**
     * Run decoder and joiner with NEMO-style greedy decoding
     * @param {ort.Tensor} encoded - Encoded features
     * @param {ort.Tensor} encodedLengths - Encoded lengths
     * @returns {Array<Hypothesis>} - Hypothesis objects with detailed results
     */
    async runDecoderJoiner(encoded, encodedLengths) {
        try {
            const batchSize = encoded.dims[0];
            const maxTime = Number(encodedLengths.data[0]);
            
            console.log(`Decoding ${maxTime} time steps with batch size ${batchSize}`);
            
            const hypotheses = [];
            
            // Process each batch
            for (let b = 0; b < batchSize; b++) {
                const hypothesis = new Hypothesis();
                
                // Initialize decoder state with correct dimensions
                console.log(`Initializing decoder states with predHidden: ${this.predHidden}`);
                let hiddenState = new ort.Tensor('float32', 
                    new Float32Array(1 * 1 * this.predHidden), 
                    [1, 1, this.predHidden]
                );
                let cellState = new ort.Tensor('float32', 
                    new Float32Array(1 * 1 * this.predHidden), 
                    [1, 1, this.predHidden]
                );
                
                // Start with blank token (<blk> is at position 1024, vocab_size from metadata is 1024)
                // Python: current_token = np.array([[self.vocab_size]], dtype=np.int32)
                // But vocab_size from metadata is 1024, and <blk> is also at 1024
                const blankTokenId = 1024; // <blk> token - this is vocab_size from Python metadata
                let currentToken = new ort.Tensor('int32', 
                    new Int32Array([blankTokenId]), 
                    [1, 1]
                );
                
                // Simple greedy decoding loop - EXACTLY matching Python implementation
                for (let t = 0; t < maxTime; t++) {
                    // Get encoder output at time t - shape: [batch, features, 1]
                    // Layout of encoded is [B, F, T] with flatten index: b*F*T + f*T + t
                    const F = encoded.dims[1];
                    const T = maxTime;
                    const encOutData = new Float32Array(F);
                    const base = b * F * T;
                    for (let f = 0; f < F; f++) {
                        encOutData[f] = encoded.data[base + f * T + t];
                    }
                    const encOut = new ort.Tensor('float32', encOutData, [1, encoded.dims[1], 1]);
                    
                    // Run decoder with current token
                    const targetLength = new ort.Tensor('int32', new Int32Array([1]), [1]);
                    
                    const decoderFeeds = {};
                    decoderFeeds[this.decoderSession.inputNames[0]] = currentToken;
                    decoderFeeds[this.decoderSession.inputNames[1]] = targetLength;
                    decoderFeeds[this.decoderSession.inputNames[2]] = hiddenState;
                    decoderFeeds[this.decoderSession.inputNames[3]] = cellState;
                    
                    const decoderResults = await this.decoderSession.run(decoderFeeds);
                    const predOut = decoderResults[this.decoderSession.outputNames[0]];
                    hiddenState = decoderResults[this.decoderSession.outputNames[2]];
                    cellState = decoderResults[this.decoderSession.outputNames[3]];
                    
                    // Run joiner
                    const joinerFeeds = {};
                    joinerFeeds[this.joinerSession.inputNames[0]] = encOut;
                    joinerFeeds[this.joinerSession.inputNames[1]] = predOut;
                    
                    const joinerResults = await this.joinerSession.run(joinerFeeds);
                    const jointOut = joinerResults[this.joinerSession.outputNames[0]];
                    
                    // Get prediction (argmax) - matching Python: joint_out.squeeze() then np.argmax
                    const logits = Array.from(jointOut.data);
                    let maxIdx = 0;
                    let maxLogit = logits[0];
                    for (let i = 1; i < logits.length; i++) {
                        if (logits[i] > maxLogit) {
                            maxLogit = logits[i];
                            maxIdx = i;
                        }
                    }
                    
                    // Store alignments if enabled
                    if (this.preserveAlignments) {
                        hypothesis.alignments.push({
                            timestep: t,
                            logits: logits,
                            predictedToken: maxIdx
                        });
                    }
                    
                    // Python logic: if pred_token_idx < self.vocab_size (not blank token)
                    // vocab_size from Python metadata is 1024, so < 1024 means not blank
                    if (maxIdx < 1024) { // Not blank token (blank is 1024)
                        // Non-blank token - add to hypothesis
                        hypothesis.addToken(maxIdx, maxLogit, t);
                        
                        // Update current token for next prediction
                        currentToken = new ort.Tensor('int32', 
                            new Int32Array([maxIdx]), 
                            [1, 1]
                        );
                    }
                    // If blank token (maxIdx >= this.vocabSize), don't add to predictions, keep current token
                }
                
                // Convert token sequence to text
                hypothesis.text = this.tokensToText(hypothesis.y_sequence);
                
                // Calculate average confidence
                if (hypothesis.confidence.length > 0) {
                    const avgConfidence = hypothesis.confidence.reduce((a, b) => a + b, 0) / hypothesis.confidence.length;
                    console.log(`Average confidence: ${avgConfidence.toFixed(4)}`);
                }
                
                hypotheses.push(hypothesis);
            }
            
            return hypotheses;
        } catch (error) {
            console.error('Error running decoder/joiner:', error);
            throw error;
        }
    }
    
    /**
     * Convert token IDs to text with proper cleanup (NEMO style)
     * @param {Array<number>} tokens - Token IDs
     * @returns {string} - Cleaned text
     */
    tokensToText(tokens) {
        let text = '';
        
        for (const tokenId of tokens) {
            if (tokenId < this.vocabulary.length) {
                text += this.vocabulary[tokenId];
            }
        }
        
        // Clean up SentencePiece tokens
        text = text.replace(/▁/g, ' ');  // Replace SentencePiece spaces
        text = text.replace(/^\s+|\s+$/g, '');  // Trim whitespace
        text = text.replace(/\s+/g, ' ');  // Normalize multiple spaces
        
        // Remove special tokens if present
        text = text.replace(/<unk>/g, '');
        text = text.replace(/<pad>/g, '');
        text = text.replace(/<s>/g, '');
        text = text.replace(/<\/s>/g, '');
        
        return text;
    }
    
    /**
     * Filter hypotheses based on confidence threshold
     * @param {Array<Hypothesis>} hypotheses - Input hypotheses
     * @returns {Array<Hypothesis>} - Filtered hypotheses
     */
    filterByConfidence(hypotheses) {
        return hypotheses.filter(hyp => {
            if (hyp.confidence.length === 0) return true;
            const avgConfidence = hyp.confidence.reduce((a, b) => a + b, 0) / hyp.confidence.length;
            return avgConfidence >= this.confidenceThreshold;
        });
    }
    
    /**
     * Transcribe audio file with NEMO-style result matching
     * @param {string} audioPath - Path to audio file
     * @param {Object} options - Transcription options
     * @returns {Object} - Detailed transcription results
     */
    async transcribe(audioPath, options = {}) {
        try {
            const {
                returnHypothesis = false,
                returnConfidence = false,
                returnAlignments = false,
                confidenceThreshold = null
            } = options;
            
            console.log(`Preprocessing audio: ${audioPath}`);
            
            // Preprocess audio
            const { processedSignal, processedLength, shape } = await preprocessAudio(audioPath, this.sampleRate);
            
            console.log(`Running encoder...`);
            
            // Run encoder
            const { encoded, encodedLengths } = await this.runEncoder(processedSignal, processedLength, shape);
            
            console.log(`Running decoder and joiner with NEMO-style matching...`);
            
            // Run decoder and joiner
            let hypotheses = await this.runDecoderJoiner(encoded, encodedLengths);
            
            // Filter by confidence if threshold is provided
            if (confidenceThreshold !== null) {
                const originalThreshold = this.confidenceThreshold;
                this.confidenceThreshold = confidenceThreshold;
                hypotheses = this.filterByConfidence(hypotheses);
                this.confidenceThreshold = originalThreshold;
            }
            
            // Get best hypothesis (first one for greedy decoding)
            const bestHypothesis = hypotheses[0];
            
            if (!bestHypothesis) {
                console.warn('No valid hypothesis found');
                return returnHypothesis ? null : "";
            }
            
            // Log detailed results
            console.log(`\n=== NEMO-Style Transcription Results ===`);
            console.log(`Text: "${bestHypothesis.text}"`);
            console.log(`Tokens: [${bestHypothesis.y_sequence.join(', ')}]`);
            console.log(`Score: ${bestHypothesis.score.toFixed(4)}`);
            
            if (bestHypothesis.confidence.length > 0) {
                const avgConfidence = bestHypothesis.confidence.reduce((a, b) => a + b, 0) / bestHypothesis.confidence.length;
                const minConfidence = Math.min(...bestHypothesis.confidence);
                const maxConfidence = Math.max(...bestHypothesis.confidence);
                console.log(`Confidence: avg=${avgConfidence.toFixed(4)}, min=${minConfidence.toFixed(4)}, max=${maxConfidence.toFixed(4)}`);
            }
            
            if (this.preserveAlignments && bestHypothesis.alignments.length > 0) {
                console.log(`Alignments: ${bestHypothesis.alignments.length} frames`);
            }
            
            console.log(`========================================\n`);
            
            // Return based on options
            if (returnHypothesis) {
                // Return full hypothesis object
                const result = {
                    hypothesis: bestHypothesis,
                    text: bestHypothesis.text,
                    score: bestHypothesis.score,
                    tokens: bestHypothesis.y_sequence
                };
                
                if (returnConfidence && bestHypothesis.confidence.length > 0) {
                    result.confidence = {
                        average: bestHypothesis.confidence.reduce((a, b) => a + b, 0) / bestHypothesis.confidence.length,
                        perToken: bestHypothesis.confidence,
                        min: Math.min(...bestHypothesis.confidence),
                        max: Math.max(...bestHypothesis.confidence)
                    };
                }
                
                if (returnAlignments && bestHypothesis.alignments.length > 0) {
                    result.alignments = bestHypothesis.alignments;
                    result.timesteps = bestHypothesis.timestep;
                }
                
                return result;
            } else {
                // Return just the text (backward compatibility)
                return bestHypothesis.text;
            }
            
        } catch (error) {
            console.error('Error during transcription:', error);
            throw error;
        }
    }
    
    /**
     * Transcribe with detailed analysis (NEMO-compatible)
     * @param {string} audioPath - Path to audio file
     * @returns {Object} - Complete analysis results
     */
    async transcribeWithAnalysis(audioPath) {
        return await this.transcribe(audioPath, {
            returnHypothesis: true,
            returnConfidence: true,
            returnAlignments: true
        });
    }
}

/**
 * Main function
 */
async function main() {
    const argv = yargs
        .option('model_dir', {
            type: 'string',
            default: '/diyoData/experiments/knowledgedistill/model',
            description: 'Directory containing ONNX models'
        })
        .option('tokens_file', {
            type: 'string',
            default: '/diyoData/experiments/knowledgedistill/model/tokens.txt',
            description: 'Path to tokens.txt file'
        })
        .option('audio_file', {
            type: 'string',
            demandOption: true,
            description: 'Path to audio file to transcribe'
        })
        .help()
        .argv;
    
    try {
        // Check if model files exist
        const modelFiles = ['encoder.onnx', 'decoder.onnx', 'joiner.onnx'];
        const missingFiles = [];
        
        for (const file of modelFiles) {
            const filePath = path.join(argv.model_dir, file);
            if (!await fs.pathExists(filePath)) {
                missingFiles.push(file);
            }
        }
        
        if (missingFiles.length > 0) {
            console.error(`Error: Missing model files in ${argv.model_dir}: ${missingFiles.join(', ')}`);
            console.error('Please run the export script first to generate the model files.');
            process.exit(1);
        }
        
        if (!await fs.pathExists(argv.tokens_file)) {
            console.error(`Error: Tokens file not found: ${argv.tokens_file}`);
            process.exit(1);
        }
        
        if (!await fs.pathExists(argv.audio_file)) {
            console.error(`Error: Audio file not found: ${argv.audio_file}`);
            process.exit(1);
        }
        
        console.log(`Loading models from: ${argv.model_dir}`);
        console.log(`Using tokens file: ${argv.tokens_file}`);
        
        // Initialize inference
        const inference = new ONNXRNNTInference(argv.model_dir, argv.tokens_file);
        await inference.initialize();
        
        // Transcribe audio
        console.log(`Transcribing: ${argv.audio_file}`);
        const transcription = await inference.transcribe(argv.audio_file);
        console.log(`Transcription: ${transcription}`);
        
    } catch (error) {
        console.error('Error:', error);
        process.exit(1);
    }
}

// Run main function if this script is executed directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = { ONNXRNNTInference, Hypothesis, loadVocabulary, preprocessAudio, NeMoStyleAudioPreprocessor };
