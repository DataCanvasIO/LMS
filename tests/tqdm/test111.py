print(ord('▍'))
print("==")
print(chr(ord('█')+1))
print(chr(ord('█')+2))
print(chr(ord('█')+3))
print(chr(ord('█')+4))
print(chr(ord('█')+5))
print(chr(ord('█')+6))
print(chr(ord('█')+7))
print("==")
# print(int.from_bytes(b'█','little'))
print(int.from_bytes(b'\x89','little'))
print(int.from_bytes(b'\x96','little'))
print(int.from_bytes(b'\x96','big'))
print(int.from_bytes(b'\x8e','little'))
print(int.from_bytes(b'\x8f','little'))

print("=====")

print(b'\x8e')

print(len(b'\x8e'))

print((8*16) +14)

print(b'\x8e'.decode('cp437'))
print(b'\x89'.decode('cp437'))
print(b'\x96'.decode('cp437'))

print(bytes((142,)).decode('cp437'))



print('█')
print(ord('█'))
print(bytes((219,)).decode('cp437'))
print(u"\u2588")