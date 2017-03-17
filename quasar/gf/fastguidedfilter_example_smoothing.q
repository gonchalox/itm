{!author name="Gonzalo Luzardo"}
{!doc category="Image Processing/Filters"}

% example: edge-preserving smoothing
import "fastguidedfilter.q"
import"Quasar.UI.dll"

I = imread("images/cat.bmp") / 255; % this image is too small to fully unleash the power of Fast GF
p = I;
r = 4;
 
eps=0.1^2;
tic();
q1 = fastguidedfilter(I[:,:,0], p[:,:,0], r, eps);
toc();

eps = 0.2^2;
tic();
q2 = fastguidedfilter(I[:,:,0], p[:,:,0], r, eps);
toc();

eps = 0.4^2;
tic();
q3 = fastguidedfilter(I[:,:,0], p[:,:,0], r, eps);
toc();


frm = form("Example Smoothing")
frm.width = 1024
frm.height = 500

disp = frm.add_display()

disp.subplot(1,4,0);
h1=disp.imshow(I,[0, 1]);

disp.subplot(1,4,1);
h2=disp.imshow(q1,[0, 1]);

disp.subplot(1,4,2);
h3=disp.imshow(q2,[0, 1]);

disp.subplot(1,4,3);
h4=disp.imshow(q3,[0, 1]);

h1.connect(h2);
h1.connect(h3);
h1.connect(h4);

frm.wait()
