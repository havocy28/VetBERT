# VetBERT
Brian Hur, b.hur@unimelb.edu.au

This tool enables the classification of the disease syndrome in a veterinary clinical notes.

The VetBERT model is a contextualized language model pretrained on over 15 million clinical documents and can be trained for a variety of tasks.

The classifier model implements VetBERT as described in the paper which can be used to classify the disease syndrome in a veterinary clinical note.

'''
@inproceedings{hur2020domain,
  title={Domain Adaptation and Instance Selection for Disease Syndrome Classification over Veterinary Clinical Notes},
  author={Hur, Brian and Baldwin, Timothy and Verspoor, Karin and Hardefeldt, Laura and Gilkerson, James},
  booktitle={Proceedings of the 19th SIGBioMed Workshop on Biomedical Language Processing},
  pages={156--166},
  year={2020}
}
'''

##Instructions

To run, install the requirements


Download the zipped VETBERT model [here](https://drive.google.com/file/d/1FwBJ6L2iQ3YUpCLgjaFFOwZDrKwIgXoj/view?usp=sharing)

Download the zipped trained classifier [here](https://drive.google.com/file/d/1lQgtbMeSo4KrYrGrqH6g94U_Lv86ugWK/view?usp=sharing)

unzip the folders contained in the files in the same file that the scripts are being ran.

ensure you have python 3.6 or higher running.  

pip install requirements.txt

to perform test classification run:

python vetbert_classify_demo.py ./input/clinical_notes.xls

If test successful, you should see the output results and there should be a file in the folder:

./output/predicted_outputs.xls

Please comment or message me if you have any questions or run into any issues.

