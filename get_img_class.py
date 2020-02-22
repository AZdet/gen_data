from pathlib import Path
import matplotlib.pyplot as plt
from generate_data import getBBoxFromMask

folder = 'full_mask'
p = Path(folder)
# car_id = []
# person_id = []
with open('car.txt', 'w+') as car, open('person.txt', 'w+') as person:
    for img_path in p.glob('*.png'):
        img = plt.imread(str(img_path))
        bbox = getBBoxFromMask(img[:,:,0])
        w = bbox[1,0] - bbox[0,0]
        h = bbox[1,1] - bbox[0,1]
        if w > h: 
            car.write(str(img_path)+'\n')
        else:
            person.write(str(img_path)+'\n')




