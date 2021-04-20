import sys
import json

def replace_folder_root(path, new_root):
    path_split = path.split("/")
    path_split[0] = new_root
    return "/".join(path_split)


if __name__ == '__main__':
    reparsed_lines = []
    odgt_path = sys.argv[1]
    reparsed_odgt_path = sys.argv[2]
    remask_folder_root_name = sys.argv[3]

    with open(odgt_path, "r") as odgt_file:
        for line in odgt_file:
            image_json = json.loads(line)
            fpath_img = image_json["fpath_img"]
            fpath_segm = image_json["fpath_segm"]

            reparsed_fpath_img = replace_folder_root(fpath_img, remask_folder_root_name)
            reparsed_fpath_segm = replace_folder_root(fpath_segm, remask_folder_root_name)

            image_json["fpath_img"] = reparsed_fpath_img
            image_json["fpath_segm"] = reparsed_fpath_segm

            reparsed_lines.append(json.dumps(image_json) + "\n")
    
    with open(reparsed_odgt_path, "w+") as reparsed_odgt_file:
        for line in reparsed_lines:
            reparsed_odgt_file.write(line)

    print(f"reparsed {odgt_path} to {reparsed_odgt_path}")
