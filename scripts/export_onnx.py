
from nemo.collections.asr.models import EncDecRNNTBPEModel

nemo_model_path = "/diyoData/experiments/knowledgedistill/experiments/without_teacher/2025-07-02_07-35-55/checkpoints/without_teacher.nemo"


model = EncDecRNNTBPEModel.restore_from(nemo_model_path)
model.eval()
model.to('cuda')  # or to('cpu') if you don't have GPU

# exporting pre-trained model to ONNX file for deployment.
model.export('onnx_export/teacher_model.onnx')

 
 