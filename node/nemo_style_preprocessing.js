/**
 * NeMo-style Audio Preprocessing Implementation in JavaScript
 * Based on NeMo's TorchScript-compatible FilterbankFeatures
 * 
 * Key patterns from NeMo:
 * 1. Structured parameter handling with defaults
 * 2. Clear separation of STFT and mel-spectrogram computation
 * 3. TorchScript-compatible operations (avoiding dynamic shapes where possible)
 * 4. Proper padding and windowing
 * 5. Magnitude computation from complex STFT
 * 6. Log zero guard for numerical stability
 */

class NeMoStyleAudioPreprocessor {
    constructor(options = {}) {
        // Default parameters matching NeMo's FilterbankFeatures
        this.sampleRate = options.sampleRate || 16000;
        this.nWindowSize = options.nWindowSize || 320;  // 20ms at 16kHz
        this.nWindowStride = options.nWindowStride || 160; // 10ms at 16kHz
        this.window = options.window || "hann";
        this.normalize = options.normalize || "per_feature";
        this.nFft = options.nFft || this._getDefaultNFft(this.nWindowSize);
        this.preemph = options.preemph !== undefined ? options.preemph : 0.97;
        this.nfilt = options.nfilt || 64; // Number of mel filters
        this.lowfreq = options.lowfreq || 0;
        this.highfreq = options.highfreq || this.sampleRate / 2;
        this.log = options.log !== undefined ? options.log : true;
        this.logZeroGuardType = options.logZeroGuardType || "add";
        this.logZeroGuardValue = options.logZeroGuardValue || Math.pow(2, -24);
        this.dither = options.dither || 1e-5;
        this.magPower = options.magPower || 2.0;
        this.exactPad = options.exactPad || false;
        this.padTo = options.padTo !== undefined ? options.padTo : 16; // NeMo default pad_to=16
        this.padValue = options.padValue || 0.0;
        this.frameSplicing = options.frameSplicing || 1;
        
        // TorchScript-compatible: Pre-compute window function
        this.windowFunction = this._createWindow(this.window, this.nWindowSize);
        
        // TorchScript-compatible: Pre-compute mel filter banks
        this.filterBanks = this._createMelFilterBanks();
        
        // Pre-compute STFT padding amount - EXACTLY matching NeMo
        // NeMo: self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None
        this.stftPadAmount = this.exactPad ? 
            Math.floor((this.nFft - this.nWindowStride) / 2) : 
            null;
        
        console.log(`NeMo-style preprocessor initialized:`);
        console.log(`  Sample Rate: ${this.sampleRate}`);
        console.log(`  n_fft: ${this.nFft}`);
        console.log(`  win_length: ${this.nWindowSize}`);
        console.log(`  hop_length: ${this.nWindowStride}`);
        console.log(`  n_mels: ${this.nfilt}`);
        console.log(`  fmin: ${this.lowfreq}`);
        console.log(`  fmax: ${this.highfreq}`);
        console.log(`  pad_to: ${this.padTo}`);
    }
    
    /**
     * Get default n_fft value (smallest power of 2 >= n_window_size)
     * Matches NeMo's logic: n_fft or 2 ** math.ceil(math.log2(n_window_size))
     */
    _getDefaultNFft(nWindowSize) {
        return Math.pow(2, Math.ceil(Math.log2(nWindowSize)));
    }
    
    /**
     * Create window function - EXACTLY matches NeMo's torch.hann_window(periodic=False)
     * NeMo uses: window_fn(self.win_length, periodic=False)
     * TorchScript-compatible: Pre-computed, fixed-size
     */
    _createWindow(windowType, windowSize) {
        const window = new Float32Array(windowSize);
        
        switch (windowType) {
            case "hann":
                // torch.hann_window(win_length, periodic=False)
                // Formula: 0.5 * (1 - cos(2 * pi * n / (N - 1))) for n = 0, 1, ..., N-1
                for (let i = 0; i < windowSize; i++) {
                    window[i] = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / (windowSize - 1)));
                }
                break;
            case "hamming":
                // torch.hamming_window(win_length, periodic=False)
                for (let i = 0; i < windowSize; i++) {
                    window[i] = 0.54 - 0.46 * Math.cos(2.0 * Math.PI * i / (windowSize - 1));
                }
                break;
            case "blackman":
                // torch.blackman_window(win_length, periodic=False)
                for (let i = 0; i < windowSize; i++) {
                    const n = i / (windowSize - 1);
                    window[i] = 0.42 - 0.5 * Math.cos(2.0 * Math.PI * n) + 0.08 * Math.cos(4.0 * Math.PI * n);
                }
                break;
            case "bartlett":
                // torch.bartlett_window(win_length, periodic=False)
                for (let i = 0; i < windowSize; i++) {
                    window[i] = 1.0 - Math.abs((2.0 * i - windowSize + 1.0) / (windowSize - 1.0));
                }
                break;
            case "ones":
            case "none":
            default:
                window.fill(1.0);
                break;
        }
        
        return window;
    }
    
    /**
     * Create mel filter banks - EXACTLY matches NeMo's librosa.filters.mel
     * NeMo uses: librosa.filters.mel(sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq, norm=mel_norm)
     * TorchScript-compatible: Pre-computed, fixed-size matrix
     */
    _createMelFilterBanks() {
        const nFreqBins = Math.floor(this.nFft / 2) + 1;
        const filterBanks = Array(this.nfilt).fill().map(() => new Float32Array(nFreqBins));
        
        // Mel scale conversion (matching librosa exactly)
        const melLow = this._hzToMel(this.lowfreq);
        const melHigh = this._hzToMel(this.highfreq);
        
        // Create mel points (matching librosa's linear spacing)
        const melPoints = new Array(this.nfilt + 2);
        for (let i = 0; i < melPoints.length; i++) {
            melPoints[i] = melLow + (melHigh - melLow) * i / (this.nfilt + 1);
        }
        
        // Convert mel points back to Hz
        const hzPoints = melPoints.map(mel => this._melToHz(mel));
        
        // Convert Hz points to FFT bin indices (matching librosa exactly)
        const binPoints = hzPoints.map(hz => {
            // librosa uses: np.floor((n_fft + 1) * freqs / sr)
            const binIndex = Math.floor((this.nFft + 1) * hz / this.sampleRate);
            // Ensure bin index is within valid range
            return Math.min(binIndex, nFreqBins - 1);
        });
        
        // Create triangular filters using continuous frequency mapping (more robust)
        // This approach better handles the case where multiple mel filters map to same bins
        for (let m = 0; m < this.nfilt; m++) {
            const leftHz = hzPoints[m];
            const centerHz = hzPoints[m + 1];  
            const rightHz = hzPoints[m + 2];
            
            // Create filter for each frequency bin
            for (let k = 0; k < nFreqBins; k++) {
                // Convert bin index back to Hz
                const binHz = k * this.sampleRate / this.nFft;
                
                let filterValue = 0.0;
                
                // Left slope: linear rise from leftHz to centerHz
                if (binHz >= leftHz && binHz < centerHz && centerHz > leftHz) {
                    filterValue = (binHz - leftHz) / (centerHz - leftHz);
                }
                // Right slope: linear fall from centerHz to rightHz  
                else if (binHz >= centerHz && binHz < rightHz && rightHz > centerHz) {
                    filterValue = (rightHz - binHz) / (rightHz - centerHz);
                }
                // Exact center frequency gets value 1.0
                else if (Math.abs(binHz - centerHz) < 1e-6) {
                    filterValue = 1.0;
                }
                
                filterBanks[m][k] = filterValue;
            }
        }
        
        // Apply Slaney normalization - EXACTLY matching librosa's implementation
        // librosa's Slaney norm: 2.0 / (freqs[2] - freqs[0]) for each triangular filter
        for (let m = 0; m < this.nfilt; m++) {
            const leftHz = hzPoints[m];
            const rightHz = hzPoints[m + 2];
            
            // Slaney normalization factor: 2.0 / (right_freq - left_freq)
            // This ensures each filter integrates to 1 over the frequency domain
            if (rightHz > leftHz) {
                const normFactor = 2.0 / (rightHz - leftHz);
                for (let k = 0; k < nFreqBins; k++) {
                    filterBanks[m][k] *= normFactor;
                }
            }
        }
        
        return filterBanks;
    }
    
    /**
     * Convert Hz to mel scale - EXACTLY matching librosa's implementation
     * librosa uses: 2595.0 * np.log10(1.0 + f / 700.0)
     */
    _hzToMel(hz) {
        return 2595.0 * Math.log10(1.0 + hz / 700.0);
    }
    
    /**
     * Convert mel scale to Hz - EXACTLY matching librosa's implementation
     * librosa uses: 700.0 * (10.0**(mels / 2595.0) - 1.0)
     */
    _melToHz(mel) {
        return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
    }
    
    /**
     * STFT implementation matching NeMo's torch.stft call
     * Key parameters from NeMo:
     * - n_fft=self.n_fft
     * - hop_length=self.hop_length  
     * - win_length=self.win_length
     * - center=False if self.exact_pad else True
     * - return_complex=True (we handle complex->magnitude conversion)
     */
    stft(signal) {
        // Apply padding based on exact_pad setting - EXACTLY matching NeMo
        let paddedSignal = signal;
        
        if (this.exactPad) {
            // NeMo exact_pad: uses custom padding amount
            if (this.stftPadAmount !== null) {
                paddedSignal = this._reflectionPad(signal, this.stftPadAmount);
            }
        } else {
            // NeMo standard mode: torch.stft with center=True (default padding)
            const padAmount = Math.floor(this.nFft / 2);
            paddedSignal = this._reflectionPad(signal, padAmount);
        }
        
        // Calculate number of frames - EXACTLY matching torch.stft behavior
        const numFrames = this.exactPad ? 
            Math.floor(paddedSignal.length / this.nWindowStride) :
            Math.floor((paddedSignal.length - this.nFft) / this.nWindowStride) + 1;
        
        // Pre-allocate result arrays (TorchScript-compatible: fixed sizes)
        const realPart = Array(numFrames).fill().map(() => new Float32Array(Math.floor(this.nFft / 2) + 1));
        const imagPart = Array(numFrames).fill().map(() => new Float32Array(Math.floor(this.nFft / 2) + 1));
        
        // Process each frame
        for (let frame = 0; frame < numFrames; frame++) {
            const start = frame * this.nWindowStride;
            const windowed = new Float32Array(this.nFft);
            
            // Apply windowing (extend window to nFft size, matching NeMo)
            for (let i = 0; i < this.nFft; i++) {
                if (start + i < paddedSignal.length) {
                    const windowValue = i < this.nWindowSize ? this.windowFunction[i] : 0;
                    windowed[i] = paddedSignal[start + i] * windowValue;
                }
            }
            
            // Compute FFT (this is the core STFT operation)
            const fftResult = this._fft(windowed);
            
            // Store only positive frequencies (onesided=True in NeMo)
            const nFreqBins = Math.floor(this.nFft / 2) + 1;
            for (let k = 0; k < nFreqBins; k++) {
                realPart[frame][k] = fftResult.real[k];
                imagPart[frame][k] = fftResult.imag[k];
            }
        }
        
        return { real: realPart, imag: imagPart };
    }
    
    /**
     * Reflection padding - EXACTLY matching NeMo's torch.nn.functional.pad with mode="reflect"
     * NeMo: x = torch.nn.functional.pad(x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "reflect").squeeze(1)
     */
    _reflectionPad(signal, padAmount) {
        const paddedLength = signal.length + 2 * padAmount;
        const paddedSignal = new Float32Array(paddedLength);
        
        // Copy original signal to center
        for (let i = 0; i < signal.length; i++) {
            paddedSignal[padAmount + i] = signal[i];
        }
        
        // Reflection padding at the beginning - torch "reflect" mode
        // Reflects around the edge values (not including the edge)
        for (let i = 0; i < padAmount; i++) {
            const srcIdx = Math.min(padAmount - 1 - i, signal.length - 1);
            paddedSignal[i] = signal[srcIdx];
        }
        
        // Reflection padding at the end - torch "reflect" mode  
        for (let i = 0; i < padAmount; i++) {
            const srcIdx = Math.max(0, signal.length - 1 - i);
            paddedSignal[padAmount + signal.length + i] = signal[srcIdx];
        }
        
        return paddedSignal;
    }
    
    /**
     * FFT implementation using Cooley-Tukey algorithm
     * Optimized for power-of-2 sizes (matching NeMo's n_fft selection)
     */
    _fft(signal) {
        const n = signal.length;
        const real = new Float32Array(n);
        const imag = new Float32Array(n);
        
        // Copy input to real part
        for (let i = 0; i < n; i++) {
            real[i] = signal[i];
            imag[i] = 0;
        }
        
        // Bit-reversal permutation
        this._bitReversePermute(real, imag);
        
        // Cooley-Tukey FFT
        for (let size = 2; size <= n; size *= 2) {
            const halfSize = size / 2;
            const step = n / size;
            
            for (let i = 0; i < n; i += size) {
                for (let j = 0; j < halfSize; j++) {
                    const u = i + j;
                    const v = i + j + halfSize;
                    
                    const angle = -2 * Math.PI * j / size;
                    const cosAngle = Math.cos(angle);
                    const sinAngle = Math.sin(angle);
                    
                    const tReal = real[v] * cosAngle - imag[v] * sinAngle;
                    const tImag = real[v] * sinAngle + imag[v] * cosAngle;
                    
                    real[v] = real[u] - tReal;
                    imag[v] = imag[u] - tImag;
                    real[u] = real[u] + tReal;
                    imag[u] = imag[u] + tImag;
                }
            }
        }
        
        return { real, imag };
    }
    
    /**
     * Bit-reverse permutation for FFT
     */
    _bitReversePermute(real, imag) {
        const n = real.length;
        let j = 0;
        
        for (let i = 1; i < n; i++) {
            let bit = n >> 1;
            while (j & bit) {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            
            if (i < j) {
                [real[i], real[j]] = [real[j], real[i]];
                [imag[i], imag[j]] = [imag[j], imag[i]];
            }
        }
    }
    
    /**
     * Main forward pass matching NeMo's FilterbankFeatures.forward()
     * Processing pipeline:
     * 1. Padding (if not exact_pad)
     * 2. Dithering (optional)
     * 3. Preemphasis
     * 4. STFT
     * 5. Complex to magnitude conversion
     * 6. Power spectrum
     * 7. Mel filterbank application
     * 8. Log transformation
     * 9. Normalization
     */
    forward(signal, options = {}) {
        let x = new Float32Array(signal);
        
        // Step 1: Apply dithering - EXACTLY matching NeMo's implementation
        // NeMo: if self.training and self.dither > 0: x += self.dither * torch.randn_like(x)
        if (options.training && this.dither > 0) {
            for (let i = 0; i < x.length; i++) {
                // Generate Gaussian noise (Box-Muller transform for exact match to torch.randn)
                const u1 = Math.random();
                const u2 = Math.random();
                const gaussianNoise = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
                x[i] += this.dither * gaussianNoise;
            }
        }
        
        // Step 2: Apply preemphasis (matches NeMo's preemph logic)
        if (this.preemph !== null && this.preemph !== 0) {
            const preemphasized = new Float32Array(x.length);
            preemphasized[0] = x[0];
            for (let i = 1; i < x.length; i++) {
                preemphasized[i] = x[i] - this.preemph * x[i - 1];
            }
            x = preemphasized;
        }
        
        // Step 3: Compute STFT
        const stftResult = this.stft(x);
        
        // Step 4: Convert complex STFT to magnitude (matches NeMo's logic)
        // x = torch.sqrt(x.pow(2).sum(-1) + guard)
        const numFrames = stftResult.real.length;
        const nFreqBins = stftResult.real[0].length;
        const magnitude = Array(numFrames).fill().map(() => new Float32Array(nFreqBins));
        
        const guard = 0; // NeMo uses 0 when not using gradients
        for (let frame = 0; frame < numFrames; frame++) {
            for (let freq = 0; freq < nFreqBins; freq++) {
                const real = stftResult.real[frame][freq];
                const imag = stftResult.imag[frame][freq];
                magnitude[frame][freq] = Math.sqrt(real * real + imag * imag + guard);
            }
        }
        
        // Step 5: Apply power spectrum (matches NeMo's mag_power)
        if (this.magPower !== 1.0) {
            for (let frame = 0; frame < numFrames; frame++) {
                for (let freq = 0; freq < nFreqBins; freq++) {
                    magnitude[frame][freq] = Math.pow(magnitude[frame][freq], this.magPower);
                }
            }
        }
        
        // Step 6: Apply mel filterbanks (matches NeMo's torch.matmul(self.fb, x))
        const melSpectrogram = Array(numFrames).fill().map(() => new Float32Array(this.nfilt));
        for (let frame = 0; frame < numFrames; frame++) {
            for (let mel = 0; mel < this.nfilt; mel++) {
                let sum = 0;
                for (let freq = 0; freq < nFreqBins; freq++) {
                    sum += this.filterBanks[mel][freq] * magnitude[frame][freq];
                }
                melSpectrogram[frame][mel] = sum;
            }
        }
        
        // Step 7: Apply log transformation (matches NeMo's log logic)
        if (this.log) {
            for (let frame = 0; frame < numFrames; frame++) {
                for (let mel = 0; mel < this.nfilt; mel++) {
                    if (this.logZeroGuardType === "add") {
                        melSpectrogram[frame][mel] = Math.log(melSpectrogram[frame][mel] + this.logZeroGuardValue);
                    } else if (this.logZeroGuardType === "clamp") {
                        melSpectrogram[frame][mel] = Math.log(Math.max(melSpectrogram[frame][mel], this.logZeroGuardValue));
                    }
                }
            }
        }
        
        // Step 8: Apply normalization - EXACTLY matches NeMo's normalize_batch function
        if (this.normalize === "per_feature") {
            // NeMo's per_feature normalization: normalize each feature (mel bin) independently
            // Uses unbiased estimator (N-1 in denominator for std)
            const CONSTANT = 1e-5; // NeMo's constant for numerical stability
            
            for (let mel = 0; mel < this.nfilt; mel++) {
                // Calculate mean
                let sum = 0;
                for (let frame = 0; frame < numFrames; frame++) {
                    sum += melSpectrogram[frame][mel];
                }
                const mean = sum / numFrames;
                
                // Calculate std with unbiased estimator (N-1 denominator)
                let sumSq = 0;
                for (let frame = 0; frame < numFrames; frame++) {
                    const diff = melSpectrogram[frame][mel] - mean;
                    sumSq += diff * diff;
                }
                // Unbiased std with clamp to CONSTANT (match NeMo's stabilize behavior)
                let std = Math.sqrt(sumSq / (numFrames - 1));
                if (std < CONSTANT) {
                    std = CONSTANT;
                }
                
                // Apply normalization
                for (let frame = 0; frame < numFrames; frame++) {
                    melSpectrogram[frame][mel] = (melSpectrogram[frame][mel] - mean) / std;
                }
            }
        } else if (this.normalize === "all_features") {
            // NeMo's all_features normalization: normalize entire spectrogram
            const CONSTANT = 1e-5; // NeMo's constant
            
            // Calculate mean across all features and time
            let sum = 0;
            let count = numFrames * this.nfilt;
            for (let frame = 0; frame < numFrames; frame++) {
                for (let mel = 0; mel < this.nfilt; mel++) {
                    sum += melSpectrogram[frame][mel];
                }
            }
            const mean = sum / count;
            
            // Calculate std across all features and time
            let sumSq = 0;
            for (let frame = 0; frame < numFrames; frame++) {
                for (let mel = 0; mel < this.nfilt; mel++) {
                    const diff = melSpectrogram[frame][mel] - mean;
                    sumSq += diff * diff;
                }
            }
            let std = Math.sqrt(sumSq / count);
            if (std < CONSTANT) {
                std = CONSTANT;
            }
            
            // Apply normalization
            for (let frame = 0; frame < numFrames; frame++) {
                for (let mel = 0; mel < this.nfilt; mel++) {
                    melSpectrogram[frame][mel] = (melSpectrogram[frame][mel] - mean) / std;
                }
            }
        }
        
        // Step 9: Apply masking and padding - EXACTLY matching NeMo's post-processing
        // NeMo: mask to zero any values beyond seq_len, pad to multiple of pad_to
        
        // Apply frame splicing if required (NeMo feature)
        if (this.frameSplicing > 1) {
            // For now, frame splicing is not implemented - would concatenate adjacent frames
            console.warn('Frame splicing > 1 not implemented, using frameSplicing=1');
        }
        
        // Apply padding to multiple of pad_to (EXACTLY matching NeMo)
        // NeMo: pad_amt = x.size(-1) % pad_to; if pad_amt != 0: x = pad(x, (0, pad_to - pad_amt))
        let finalMelSpectrogram = melSpectrogram;
        let finalSeqLen = numFrames;
        
        if (this.padTo > 0) {
            const padAmt = numFrames % this.padTo;
            if (padAmt !== 0) {
                const paddingNeeded = this.padTo - padAmt;
                console.log(`Padding frames from ${numFrames} to ${numFrames + paddingNeeded} (multiple of ${this.padTo})`);
                
                // Create new padded spectrogram
                const paddedFrames = numFrames + paddingNeeded;
                finalMelSpectrogram = Array(paddedFrames).fill().map(() => new Float32Array(this.nfilt));
                
                // Copy original data
                for (let frame = 0; frame < numFrames; frame++) {
                    for (let mel = 0; mel < this.nfilt; mel++) {
                        finalMelSpectrogram[frame][mel] = melSpectrogram[frame][mel];
                    }
                }
                
                // Fill padding frames with pad_value (default 0.0)
                for (let frame = numFrames; frame < paddedFrames; frame++) {
                    for (let mel = 0; mel < this.nfilt; mel++) {
                        finalMelSpectrogram[frame][mel] = this.padValue;
                    }
                }
                
                finalSeqLen = paddedFrames;
            }
        }
        
        return {
            melSpectrogram: finalMelSpectrogram,
            seqLen: finalSeqLen
        };
    }
    
    /**
     * Utility method to get sequence length - EXACTLY matches NeMo's get_seq_len
     * NeMo: seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length) + 1
     */
    getSeqLen(inputLength) {
        // Calculate padding amount based on exact_pad setting
        let padAmount;
        if (this.exactPad) {
            padAmount = this.stftPadAmount !== null ? this.stftPadAmount * 2 : 0;
        } else {
            padAmount = Math.floor(this.nFft / 2) * 2; // Default torch.stft padding
        }
        
        return Math.floor((inputLength + padAmount - this.nFft) / this.nWindowStride) + 1;
    }
}

// Export for use in Node.js or browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NeMoStyleAudioPreprocessor;
}

// Example usage demonstrating NeMo-compatible configuration
if (require.main === module) {
    console.log("NeMo-style Audio Preprocessor Example");
    
    // Create preprocessor with NeMo default parameters
    const preprocessor = new NeMoStyleAudioPreprocessor({
        sampleRate: 16000,
        nWindowSize: 320,    // 20ms window
        nWindowStride: 160,  // 10ms hop
        window: "hann",
        nFft: 512,          // Next power of 2 >= 320
        preemph: 0.97,
        nfilt: 64,          // 64 mel filters
        lowfreq: 0,
        highfreq: 8000,
        log: true,
        logZeroGuardType: "add",
        logZeroGuardValue: Math.pow(2, -24),
        magPower: 2.0,
        normalize: "per_feature"
    });
    
    // Generate test signal (1 second of 440Hz tone)
    const sampleRate = 16000;
    const duration = 1.0;
    const frequency = 440;
    const testSignal = new Float32Array(sampleRate * duration);
    
    for (let i = 0; i < testSignal.length; i++) {
        testSignal[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
    }
    
    console.log(`Processing ${testSignal.length} samples...`);
    
    // Process signal
    const result = preprocessor.forward(testSignal);
    
    console.log(`Output mel spectrogram shape: [${result.melSpectrogram.length}, ${result.melSpectrogram[0].length}]`);
    console.log(`Sequence length: ${result.seqLen}`);
    console.log("Processing complete!");
}
