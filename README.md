# Evaluation of Teacher Model (600M)

LibriSpeech Train(100 hour)
WER: 0.0138
CER: 0.0041


LibriSpeech Test Clean :  
WER: 0.0182
CER: 0.0052

# Evaluation of  Model (110M): trained on LibriSpeech Train

LibriSpeech Test Clean :  

WER: 0.1123
CER: 0.0508


## KD Loss  nan problem 

Teacher joint range: -52.36 to 117.31 (very wide, 170 units)
Student joint range: -1.47 to 1.56 (very narrow, 3 units)
After temperature scaling:

Teacher: -5.24 to 11.73 (still wide)
Student: -0.15 to 0.16 (very narrow)


This creates vastly different probability distributions:

Teacher: Sharp, confident predictions (low entropy)
Student: Uniform, uncertain predictions (high entropy)



# Commands 

python train.py --config-path=./  --config-name=fast-conformer_transducer_bpe.yaml

