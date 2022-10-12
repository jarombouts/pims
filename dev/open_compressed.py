import pims
import matplotlib.pyplot as plt


pims_image_object = pims.open('./dev/12-28-53.052.seq')
f0 = pims_image_object[0]
f1 = pims_image_object[1]
fn = pims_image_object[-1]

for f in (f0, f1, fn):
    plt.imshow(f)
    plt.show()

pass