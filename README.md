# The Sunshine group presents:
## Investigating sustainability indicators of Free Open Source Software projects' (FOSS) relation to Apache Software Foundations decision-making 

This repository contains the results of research done to try to uncover patterns related to code quality and sustainability metrics for a dataset of FOSS projects that are or have been a part of the Apache Foundation or the Apache Foundations incubator.  

---

### Brute force clustering
Output of brute force 7 feature model clustering is done with the file [BruteForceClustering.py](BruteForceClustering.py) and the result is saved to [7-features-result.csv.zip](7-features-result.csv.zip)

---

### Preprocessing
Preprocessing is done with the file [PreProcessing.ipynb](PreProcessing.ipynb) which takes the files:
- [non_incubator_project_metrics.csv](non_incubator_project_metrics.csv)
- [incubator_project_metrics_graduated_retired.csv](incubator_project_metrics_graduated_retired.csv)
- [incubator_project_graduated_from_graduation.csv](incubator_project_graduated_from_graduation.csv) 

And outputs the result to a single file:
- [combined_project_status.csv](combined_project_status.csv)

---

### Research files
The rest of the files have been used as a basis of research towards the end result. They contain different angles at atempting to bruteforce optimal clustering or include avenues of research that was not included in the final methodology or findings. 
