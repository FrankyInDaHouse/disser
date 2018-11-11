CODE_LENGTH = 7

f = open('codewords.txt', 'w')

i = 0
while i < (2 ** CODE_LENGTH):
	f.write(format(i, '07b'))
	f.write(',')
	f.write(str(i))
	f.write('\n')
	i = i + 1
f.close()