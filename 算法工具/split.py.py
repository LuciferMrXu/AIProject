# encoding='utf8'
import re
def split(inputs, outputs):
    with open(inputs, encoding='utf8') as fin, open(outputs, 'w', encoding='utf8') as fo:
        for line in fin:
            line = re.sub('\n','\n\n',line)
            #print(line)
            line = re.sub(r"([，,。！!?？;；])", lambda x: "{0}\n".format(x.group(1)), line)
            pattern = re.compile(r'\n\n\n$')
            line=re.sub(pattern,r'\n\n',line)
            fo.write(line)
          
            
if __name__ == "__main__":
    split(inputs='raw.txt', outputs='test.txt')