import "imresize.q"
import "boxfilter.q"
import "linalg.q"

{!author name="Gonzalo Luzardo"}
{!doc category="Image Processing/Filters"}

function q:mat = fastguidedfilter(I : mat, p : mat, r : scalar, eps : scalar, s : scalar = 4)
    %   GUIDEDFILTER   O(1) time implementation of guided filter.
    %
    %   - guidance image: I (should be a gray-scale/single channel image)
    %   - filtering input image: p (should be a gray-scale/single channel image)
    %   - local window radius: r
    %   - regularization parameter: eps
    %   - subsampling ratio: s (try s = r/4 to s=r)
    %
    %    Coding translated from Matlab source available in http://research.microsoft.com/en-us/um/people/kahe/eccv10/
    %    Author: Gonzalo Luzardo

    I_sub:mat = imresize(I:mat, 1/s, "nearest"); % NN is often enough
    p_sub:mat = imresize(p:mat, 1/s, "nearest");
    r_sub = r / s; % make sure this is an integer

 
    assert(r_sub-int(r_sub)==0,"r/s must be an integer!!. Default value of s is 4")
       
    [hei,wid] = size(I_sub);
    
    N = boxfilter(ones(hei, wid), r_sub); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.
    mean_I = boxfilter(I_sub, r_sub) ./ N;
    mean_p = boxfilter(p_sub, r_sub) ./ N;
    mean_Ip = boxfilter(I_sub.*p_sub, r_sub) ./ N;
    cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
    
    mean_II = boxfilter(I_sub.*I_sub, r_sub) ./ N;
    var_I = mean_II - mean_I .* mean_I;
    
    a = cov_Ip ./ (var_I + eps);
    b = mean_p - a .* mean_I;
    
    mean_a = boxfilter(a, r_sub) ./ N;
    mean_b = boxfilter(b, r_sub) ./ N;
    
    mean_a:mat = imresize(mean_a, [size(I)[0], size(I)[1]], "bilinear"); % bilinear is recommended 
    mean_b:mat = imresize(mean_b, [size(I)[0], size(I)[1]], "bilinear");
    
    q = mean_a .* I + mean_b;
 end
 
 function q:mat = fastguidedfilter_color(I : cube, p : mat, r : scalar, eps : scalar, s : scalar = 4)
    %   GUIDEDFILTER_COLOR   O(1) time implementation of guided filter using a color image as the guidance.
    %
    %   - guidance image: I (should be a color (RGB) image)
    %   - filtering input image: p (should be a gray-scale/single channel image)
    %   - local window radius: r
    %   - regularization parameter: eps
    %   - subsampling ratio: s (try s = r/4 to s=r)
    %
    %    Coding translated from Matlab source available in http://research.microsoft.com/en-us/um/people/kahe/eccv10/
    %    Author: Gonzalo Luzardo
    
    I_sub:cube = imresize(I:cube, 1/s, "nearest"); % NN is often enough
    p_sub:mat = imresize(p:mat, 1/s, "nearest");
    r_sub = r / s; % make sure this is an integer
    
    assert(r_sub-int(r_sub)==0,"r/s must be an integer!!. Default value of s is 4")
    
    [hei,wid] = size(p_sub);
    
    N = boxfilter(ones(hei, wid), r_sub); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.

    mean_I_r = boxfilter(I_sub[:, :, 0], r_sub) ./ N;
    mean_I_g = boxfilter(I_sub[:, :, 1], r_sub) ./ N;
    mean_I_b = boxfilter(I_sub[:, :, 2], r_sub) ./ N;

    mean_p = boxfilter(p_sub, r_sub) ./ N;

    mean_Ip_r = boxfilter(I_sub[:, :, 0].*p_sub, r_sub) ./ N;
    mean_Ip_g = boxfilter(I_sub[:, :, 1].*p_sub, r_sub) ./ N;
    mean_Ip_b = boxfilter(I_sub[:, :, 2].*p_sub, r_sub) ./ N;

    % covariance of (I, p) in each local patch.
    cov_Ip_r = mean_Ip_r - mean_I_r .* mean_p;
    cov_Ip_g = mean_Ip_g - mean_I_g .* mean_p;
    cov_Ip_b = mean_Ip_b - mean_I_b .* mean_p;

    % variance of I in each local patch: the matrix Sigma in Eqn (14).
    % Note the variance in each local patch is a 3x3 symmetric matrix:
    %           rr, rg, rb
    %   Sigma = rg, gg, gb
    %           rb, gb, bb
    var_I_rr = boxfilter(I_sub[:, :, 0].*I_sub[:, :, 0], r_sub) ./ N - mean_I_r .*  mean_I_r; 
    var_I_rg = boxfilter(I_sub[:, :, 0].*I_sub[:, :, 1], r_sub) ./ N - mean_I_r .*  mean_I_g; 
    var_I_rb = boxfilter(I_sub[:, :, 0].*I_sub[:, :, 2], r_sub) ./ N - mean_I_r .*  mean_I_b; 
    var_I_gg = boxfilter(I_sub[:, :, 1].*I_sub[:, :, 1], r_sub) ./ N - mean_I_g .*  mean_I_g; 
    var_I_gb = boxfilter(I_sub[:, :, 1].*I_sub[:, :, 2], r_sub) ./ N - mean_I_g .*  mean_I_b; 
    var_I_bb = boxfilter(I_sub[:, :, 2].*I_sub[:, :, 2], r_sub) ./ N - mean_I_b .*  mean_I_b; 

    a = zeros(hei, wid, 3);
    for y=0..(hei-1)
        for x=0..(wid-1) 
            Sigma = [[var_I_rr[y, x], var_I_rg[y, x], var_I_rb[y, x]], 
                      [var_I_rg[y, x], var_I_gg[y, x], var_I_gb[y, x]], 
                      [var_I_rb[y, x], var_I_gb[y, x], var_I_bb[y, x]]];
            cov_Ip = [cov_Ip_r[y, x], cov_Ip_g[y, x], cov_Ip_b[y, x]];        
            a[y, x, :] = cov_Ip * inv3x3(Sigma + eps * eye(3)); 
        end
    end
    

    b = mean_p - a[:, :, 0] .* mean_I_r - a[:, :, 1] .* mean_I_g - a[:, :, 2] .* mean_I_b; % Eqn. (15) in the paper;
    mean_a = mat(size(a))
    mean_b = mat(size(b))
    mean_a[:, :, 0] = boxfilter(a[:, :, 0], r_sub)./N;
    mean_a[:, :, 1] = boxfilter(a[:, :, 1], r_sub)./N;
    mean_a[:, :, 2] = boxfilter(a[:, :, 2], r_sub)./N;
    mean_b = boxfilter(b, r_sub)./N;

    mean_a = imresize(mean_a, [size(I, 0), size(I, 1)], "bilinear"); % bilinear is recommended
    mean_b = imresize(mean_b, [size(I, 0), size(I, 1)], "bilinear");

    q1= mean_a[:, :, 0] .* I[:, :, 0] +
        mean_a[:, :, 1] .* I[:, :, 1] +
        mean_a[:, :, 2] .* I[:, :, 2] 
    q = q1 + mean_b;
end


function out:scalar = det3x3(a:mat)
    out = (a[0,0] * (a[1,1] * a[2,2] - a[2,1] * a[1,2])
           -a[1,0] * (a[0,1] * a[2,2] - a[2,1] * a[0,2])
           +a[2,0] * (a[0,1] * a[1,2] - a[1,1] * a[0,2]))  
end

function res:mat = inv3x3(m:mat)
    d = det3x3(m);
    res = mat(3,3);
    res[0,0]=m[1,1]*m[2,2]-m[1,2]*m[2,1];
    res[0,1]=-m[0,1]*m[2,2]+m[0,2]*m[2,1];
    res[0,2]=m[0,1]*m[1,2]-m[0,2]*m[1,1];  
    
    res[1,0]=-m[1,0]*m[2,2]+m[1,2]*m[2,0];
    res[1,1]=m[0,0]*m[2,2]-m[0,2]*m[2,0];
    res[1,2]=-m[0,0]*m[1,2]+m[0,2]*m[1,0];
    
    res[2,0]=m[1,0]*m[2,1]-m[1,1]*m[2,0];
    res[2,1]=-m[0,0]*m[2,1]+m[0,1]*m[2,0];
    res[2,2]=m[0,0]*m[1,1]-m[0,1]*m[1,0];
    res = res./d;
end

function [] = main()
    I = imread("lena_big.tif") / 255
    p = I
    r = 4;
    imshow(I,[0, 1])
   
    eps=0.1^2
    q1=uninit(size(I))
    tic()
    q1[:,:,0] = fastguidedfilter(I[:,:,0], p[:,:,0], r, eps)
    q1[:,:,1] = fastguidedfilter(I[:,:,1], p[:,:,1], r, eps)
    q1[:,:,2] = fastguidedfilter(I[:,:,2], p[:,:,2], r, eps)
    toc()
    imshow(q1,[0, 1]);

    eps = 0.2^2
    q2=uninit(size(I))
    tic()
    q2[:,:,0] = fastguidedfilter(I[:,:,0], p[:,:,0], r, eps)
    q2[:,:,1] = fastguidedfilter(I[:,:,1], p[:,:,1], r, eps)
    q2[:,:,2] = fastguidedfilter(I[:,:,2], p[:,:,2], r, eps)
    toc()
    imshow(q2,[0, 1])
    
    eps = 0.4^2
    q3=uninit(size(I))
    tic()
    q3[:,:,0] = fastguidedfilter(I[:,:,0], p[:,:,0], r, eps)
    q3[:,:,1] = fastguidedfilter(I[:,:,1], p[:,:,1], r, eps)
    q3[:,:,2] = fastguidedfilter(I[:,:,2], p[:,:,2], r, eps)
    toc()
    imshow(q3,[0, 1])


end
 
 
 