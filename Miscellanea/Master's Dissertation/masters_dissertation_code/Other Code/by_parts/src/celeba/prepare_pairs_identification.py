import os
from shutil import rmtree, copyfile

if(os.path.exists("pairs_to_explain")): rmtree("pairs_to_explain")
os.makedirs("pairs_to_explain")

performance_evaluation_pairs = list(filter(lambda x : x[0] != ".", os.listdir("performance_evaluation_pairs")))

for i in performance_evaluation_pairs:
    copyfile("performance_evaluation_pairs/" + i + "/1.jpg", "pairs_to_explain/" + i.split("_")[0] + "_1_" + i.split("_")[3].replace("C", "") + ".jpg")
    copyfile("performance_evaluation_pairs/" + i + "/2.jpg", "pairs_to_explain/" + i.split("_")[0] + "_2_" + i.split("_")[4].replace("C", "") + ".jpg")
