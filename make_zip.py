import os
import shutil

curr_dir = os.getcwd()
contents = os.listdir(curr_dir)
# print(f"{contents}")

for items in contents:
    if os.path.isdir(os.path.join(curr_dir, items)):
        if items == 'Parts':
            shutil.make_archive('parts', 'zip', os.path.join(curr_dir, items))
            print(f"updated ZIP")