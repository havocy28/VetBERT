# Update
---------
The VetBERT model has been converted to PyTorch and moved to HuggingFace. The pretrained model without finetuing is located at: [https://huggingface.co/havocy28/VetBERT](https://huggingface.co/havocy28/VetBERT)

The VetBERT model finetuned on the disease syndrome classification task is located at: [https://huggingface.co/havocy28/VetBERTDx](https://huggingface.co/havocy28/VetBERTDx)


# VetBERT
---------


Brian Hur, b.hur@unimelb.edu.au

VetBERT is a [BERT based](https://github.com/google-research/bert) contextualized language model pretrained on over 15 million veterinary clinical notes and can be trained to perform a variety of tasks such as the disease indicated in a veterinary clinical record.

The classifier model implements VetBERT as described in the [paper and presentation](https://www.aclweb.org/anthology/2020.bionlp-1.17/) from the BioNLP workshop @ ACL 2020 which can be used to classify the disease syndrome in a veterinary clinical note.



# Instructions

To run, install the requirements


Download the zipped VETBERT model [here](https://drive.google.com/file/d/1FwBJ6L2iQ3YUpCLgjaFFOwZDrKwIgXoj/view?usp=sharing)

Download the zipped trained classifier [here](https://drive.google.com/file/d/1lQgtbMeSo4KrYrGrqH6g94U_Lv86ugWK/view?usp=sharing)

unzip the folders contained in the files in the same file that the scripts are being ran.

ensure you have python 3.6 or higher running.  

```

pip install requirements.txt

```

to perform test classification run:

```

python vetbert_classify_demo.py ./input/clinical_notes.xls

```

If test successful, you should see the output results and there should be a file in the folder:

./output/predicted_outputs.xls

To classify your own notes, follow the format in ./input/clinical_notes.xls and save using Excel 97-2003 format.
You need to supply a dummy label if you do not have the labels and are note testing the model.  The labels that can be used
are listed in labels.txt.


The following paper should be cited if you use any of these resources:

```

@inproceedings{hur2020domain,
  title={Domain Adaptation and Instance Selection for Disease Syndrome Classification over Veterinary Clinical Notes},
  author={Hur, Brian and Baldwin, Timothy and Verspoor, Karin and Hardefeldt, Laura and Gilkerson, James},
  booktitle={Proceedings of the 19th SIGBioMed Workshop on Biomedical Language Processing},
  pages={156--166},
  year={2020}
}

```


Please comment or message me if you have any questions or run into any issues.
