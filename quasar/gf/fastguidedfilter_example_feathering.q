{!author name="Gonzalo Luzardo"}
{!doc category="Image Processing/Filters"}

% example: guided feathering
import "fastguidedfilter.q"
import "Quasar.UI.dll"

I = imread("images/toy.bmp") / 255;
p = imread("images/toy-mask.bmp") / 255;

eps = 10^-6;
r = 32
s = 16
tic();
q = fastguidedfilter_color(I, p[:,:,1], r, eps,s); %Using the Green channel
toc();

frm = form("Example Feathering")
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

