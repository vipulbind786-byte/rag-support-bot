import easyocr
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
check = {26:"t=12.5 Q+A", 30:"t=14.5 Q+A", 33:"t=16.5 Q2", 36:"t=18.0 Q2+A2", 39:"t=19.5 stress", 43:"t=21.0 stress", 46:"t=22.5 end"}
for n, label in check.items():
    res = reader.readtext(f"frames/f_{n:03d}.png")
    print(f"{label}: " + " | ".join(t[1] for t in res))
