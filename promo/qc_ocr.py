import easyocr, glob
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
picks = {1:"0-2.5 doc",3:"2.5-4.5 answers",5:"4.5-7 no cloud",9:"7-11.5 title+pipeline",13:"11.5-16 Q+A",17:"16-18.5 Q+A",20:"18.5-21.5 stress",23:"21.5-24 end"}
for n, label in picks.items():
    f = f"frames/f_{n:03d}.png"
    res = reader.readtext(f)
    txt = " | ".join(t[1] for t in res)
    print(f"{label}: {txt}")
