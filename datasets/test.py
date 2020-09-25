
import  itertools
choices=['0','1','2','3','4','5','6','7','8','9']
num_per_image=4
digits=[]
for i in itertools.permutations(choices, num_per_image):
    captcha = ''.join(i)
    digits.append(captcha)

print(len(digits))
