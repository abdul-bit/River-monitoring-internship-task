import splitfolders

input_folder = "C:/Users/Abdul/OneDrive/Desktop/Internship Task/Dataset"


splitfolders.ratio(input_folder, output="C:/Users/Abdul/OneDrive/Desktop/Internship Task/Dataset",
                   seed=42, ratio=(.9,  .1),
                   group_prefix=None)
