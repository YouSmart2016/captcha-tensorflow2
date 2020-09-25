# captcha-tensorflow2
# 彩色验证码生成
```bash
python datasets/gen_captcha.py -d --npi=4 -n 6
```
npi指生成的字符个数，n指的回合的次数，上述代码生成的图片张数10\*9\*8\*7=5040张
#  黑白验证码生成
```python
img = ImageCaptcha(width=100,height=100)
#chars指传入的字符，color指前景色，background 指背景色
im = img.create_captcha_image(chars='1234', color='white', background='black')
im.save('test.jpg')
```