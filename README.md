# Models
CRF

BiLSTM

BiLSTM-CRF

# Setup env
**pip install -r requirments.txt**

# Run demo
- Run VnCoreNLP on port :8000
  
  vncorenlp -Xmx2g "VnCoreNLP/VnCoreNLP-1.1.1.jar" -p 8000 -a "wseg,pos,ner,parse"
- Run: python3 main.py

  service run on port :1234
