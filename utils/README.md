# Utils

* ## [prepare_dataset.py](prepare_dataset.py) - downloads and prepares the dataset  
  * Downloads the SignalP5.0 dataset from their server
  * Converts the 3-line fasta format to a tsv format
  * Cleans the dataset (currently only removes sequences with length other than 70)
  * Splits the dataset into training and test sets

