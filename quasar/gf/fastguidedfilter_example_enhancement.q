{!author name="Gonzalo Luzardo"}
{!doc category="Image Processing/Filters"}

% example: detail enhancement
import "fastguidedfilter.q"
import "Quasar.UI.dll"

I = (imread("images/tulips.bmp")) / 255;
p = I;

% r/s must be an integer
r = 8;
eps = 0.1^2;

q = zeros(size(I));

tic();
q[:, :, 0] = fastguidedfilter(I[:, :, 0], p[:, :, 0], r, eps);
q[:, :, 1] = fastguidedfilter(I[:, :, 1], p[:, :, 1], r, eps);
q[:, :, 2] = fastguidedfilter(I[:, :, 2], p[:, :, 2], r, eps);
toc();

I_enhanced = (I - q) * 5 + q;

frm = form("Example Enhancement")
frm.width = 1024
frm.height = 600
disp = frm.add_display()
disp.subplot(1,3,0);
h1=disp.imshow(I,[0, 1]);
disp.subplot(1,3,1);
h2=disp.imshow(q,[0, 1]);
disp.subplot(1,3,2);
h3=disp.imshow(I_enhanced,[0, 1]);
h1.connect(h2);
h1.connect(h3);
frm.wait()
