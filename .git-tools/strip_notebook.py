import sys,json
raw=sys.stdin.buffer.read()
doc=json.loads(raw.decode("utf-8-sig"))
for cell in doc.get("cells",[]):
    if cell.get("cell_type")=="code":
        cell["outputs"]=[]
        cell["execution_count"]=None
doc.get("metadata",{}).pop("widgets",None)
json.dump(doc,sys.stdout,ensure_ascii=False,indent=1)
sys.stdout.write("\n")
