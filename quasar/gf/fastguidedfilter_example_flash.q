{!author name="Gonzalo Luzardo"}
{!doc category="Image Processing/Filters"}

% example: flash/noflash denoising
import "fastguidedfilter.q"
import"Quasar.UI.dll"

I = imread("images/cave-flash.bmp") / 255;
p = imread("images/cave-noflash.bmp") / 255;

r = 8;
eps = 0.02^2;
q = zeros(size(I));

tic();
q[:, :, 0] = fastguidedfilter(I[:, :, 0], p[:, :, 0], r, eps);
q[:, :, 1] = fastguidedfilter(I[:, :, 1], p[:, :, 1], r, eps);
q[:, :, 2] = fastguidedfilter(I[:, :, 2], p[:, :, 2], r, eps);
toc();

frm = form("Example Flash")
frm.width = 1024
frm.height = 500

disp = frm.add_display()

disp.subplot(1,3,0);
h1=disp.imshow(I,[0, 1]);

disp.subplot(1,3,1);
h2=disp.imshow(p,[0, 1]);

disp.subplot(1,3,2);
h3=disp.imshow(q,[0, 1]);

h1.connect(h2);
h1.connect(h3);

frm.wait()

