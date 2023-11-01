import json


def test():
    txt_path = r"D:\Desktop\imagenet1000_clsidx_to_labels.txt"
    save_path = r"D:\workspace\code\study\torch_classification\utils\imagenet1000.json"
    dicts = {}
    with open(txt_path, "r", encoding="utf-8") as fr:
        infos = fr.readlines()
    for idx, per_info in enumerate(infos):
        li = per_info.split(":")
        cls = li[0][1:]
        cls_name = li[1].strip("\n").strip(" ")
        if cls_name[0] == "'":
            cls_name = cls_name.split("'")[1]
        else:
            cls_name = cls_name.split('"')[1]
        print(idx, cls, cls_name)
        dicts[cls] = cls_name

    json_str = json.dumps(dicts, indent=4)
    with open(save_path, "w") as json_file:
        json_file.write(json_str)


def test02():
    with open("./utils/imagenet100.json", "r") as fr:
        infos = json.load(fr)
    reverser_infos = {v: k for k, v in infos.items()}

    json_str = json.dumps(reverser_infos, indent=4)
    with open("./utils/imagenet100_new.json", "w") as json_file:
        json_file.write(json_str)