import "Quasar.Video.dll"
import "Quasar.UI.dll"
import "Quasar.Runtime.dll"
import "Sim2HDR.dll"
import "Quasar.UI.dll"
import "immorphology.q"
import "fastguidedfilter.q"
import "inttypes.q"
import "system.q"
import "colortransform.q"
import "C:\Users\ipi\Documents\gluzardo\eotf_pq\quasar\transfer_functions.q"
import "C:\Users\ipi\Documents\gluzardo\quasar_sim2\sim2.q"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Toe, shoulder Tonemapper                                             %
% http://gpuopen.com/wp-content/uploads/2016/03/GdcVdrLottes.pdf       %
% {a:contrast,d:shoulder} shapes curve                                     %
% {b,c} anchors curve                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Linearize
function y = linearize(x)
    y=sRGB_decode(x)
end

%Delinearize
function y = delinearize(x)
    y=sRGB_encode(x)
end

%Update B and C values
function [] = updateBC(t:object)
    t.b= (-t.midIn^t.a + t.hdrMax^t.a*t.midOut)/(((t.hdrMax^t.a)^t.d-(t.midIn^t.a)^t.d) * t.midOut);
    t.c= ((t.hdrMax^t.a)^t.d*t.midIn^t.a-t.hdrMax^t.a *(t.midIn^t.a)^t.d*t.midOut)/(((t.hdrMax^t.a)^t.d-(t.midIn^t.a)^t.d)*t.midOut)
end

%Clampp values between l and h
function [y:vec3] = __device__ clamp_values(x:vec3'unchecked,l:scalar,h:scalar)
    y=uninit(size(x))
    for i=0..2
        if x[i]>h
            y[i]=h
        elseif x[i]<l
            y[i]=l  
        else
            y[i]=x[i]
        endif  
    end
end

%Apply lut
function [y:cube] = EO_LUT(x:cube'unchecked,low:cube'unchecked,high:cube'unchecked,lut:vec'unchecked)
    entries = max(size(lut))
    function [] = __kernel__ EOTF_LUT_kern(x:cube'unchecked,y:cube'unchecked,low:cube'unchecked,high:cube'unchecked,lut:vec'clamped,resol:int,pos:ivec3)
        index = floor(x[pos[0],pos[1],pos[2]]*(resol-1))
        y[pos[0],pos[1],pos[2]] = lut[index]
        low[pos[0],pos[1],pos[2]] = lut[index]-3*(lut[index]-lut[index-1])/2
        high[pos[0],pos[1],pos[2]] = lut[index]+3*(lut[index+1]-lut[index])/2
    end
    y = uninit(size(x))
    parallel_do(size(y),x,y,low,high,lut,entries,EOTF_LUT_kern)
end

%%Kernel to Apply tmo operator to Separation of Max an RGB ratio in parllel
function [y:vec3] = __device__ expandPixel(x:vec3,a:scalar,b:scalar,c:scalar,d:scalar,peak_lum:scalar)
        %Apply Separation of Max an RGB ratio
        peak=max(x)
        ratio=x*(1/peak)
        peak=(peak.^a)./(((peak.^a).^d).*b+c)
        y:vec3=peak*ratio
        y=peak_lum*y
        %Clamp and peak luminance ratio
        %output=p*clamp_values(output,0.0,1.0) %Clamp values between 0 and peak luminance
end


%         %Apply Separation of Max an RGB ratio
%            m=max(x[pos[0],pos[1],:]);
%            ratio=x[pos[0],pos[1],:]/m;
%            m = m + b*(mask[pos[0],pos[1]])
%            y[pos[0],pos[1],:]=m*ratio;
%   

%Apply lut take care of color change
function [y:cube] = EO_LUT_CF(x:cube'unchecked,low:cube'unchecked,high:cube'unchecked,params:object,lut:vec'unchecked)
    entries = max(size(lut))
    function [] = __kernel__ EO_LUT_CF_kern(x:cube'unchecked,y:cube'unchecked,low:cube'unchecked,high:cube'unchecked,a:scalar,b:scalar,c:scalar,d:scalar,p:scalar,lut:vec'clamped,resol:int,pos:ivec2)
        input = [x[pos[0],pos[1],0],x[pos[0],pos[1],1],x[pos[0],pos[1],2]]
        output=expandPixel(input,a,b,c,d,p)
        y[pos[0],pos[1],0]=output[0]
        y[pos[0],pos[1],1]=output[1]
        y[pos[0],pos[1],2]=output[2]
        %index = floor(y[pos[0],pos[1],pos[2]]*(resol-1))
        
        %low[pos[0],pos[1],pos[2]] = lut[index]-3*(lut[index]-lut[index-1])/2
        %high[pos[0],pos[1],pos[2]] = lut[index]+3*(lut[index+1]-lut[index])/2
    end
    y = zeros(size(x))
    parallel_do(size(x,0..1),x,y,low,high,params.a,params.b,params.c,params.d,params.peak_luminance,lut,entries,EO_LUT_CF_kern)
end

function [out] = __device__ expandVal(in:scalar'unchecked,a,b,c,d,p)
    out=(in.^a)./(((in.^a).^d).*b+c);
    out=p*clamp(out,1.0) %Clamp values between 0 and peak luminance
end

%Kernel to Apply tmo operator to Separation of Max an RGB ratio in parllel
function [y:vec] = getLut(x:vec,params:object)
    function []= __kernel__ getLut_kernel(x:vec'unchecked, y:vec'unchecked,a:scalar,b:scalar,c:scalar,d:scalar,p:scalar,pos:ivec2)
        input=x[pos[0],pos[1]];
        y[pos[0],pos[1]]= expandVal(input,a,b,c,d,p);
    end
    y=uninit(size(x)) 
    %Linearize the image
    %x=linearize(x)
    parallel_do(size(x),x,y,params.a,params.b,params.c,params.d,params.peak_luminance,getLut_kernel)
end

%Unmake
function [] = unmake_raw_cube(x:cube, y:cube)
    y[:,:,0..2] = x[:,:,[2,1,0]]
end

% 1D Horizontal filter kernel
function [] = __kernel__ pocs_horizontal_run(y : cube'unchecked, _
    x : cube' clamped, r : int, pos : ivec3)
    sum = 0.0
    for m=0..2*r
        sum += x[pos + [0,m-r,0]]
    end
    y[pos] = sum/(2*r+1)
end

% 1D Vertical filter kernel
function [] = __kernel__ pocs_vertical_run(y : cube'unchecked, _
    x : cube' clamped, r : int, high:cube' unchecked, low:cube'unchecked, pos : ivec3)
    sum = 0.0
    for m=0..2*r
        sum += x[pos + [m-r,0,0]]
    end
    y[pos] = min(max(sum/(2*r+1),low[pos]),high[pos])
end

%Get luminance using HSL color space
function [lum] = __device__ getluminance(c : vec3)
    cmax = max(c)
    cmin = min(c)
    lum = 0.5 * (cmin + cmax)  % lightness
end

%Get lightness using HSL color space
function [lum:mat] = getLightnessImageHSL(image : cube)
    function [] = __kernel__ getLuma_kernel(x:cube'unchecked,y:mat'unchecked,pos:ivec2)
       cmax = max([x[pos[0],pos[1],0],x[pos[0],pos[1],1],x[pos[0],pos[1],2]])
       cmin = min([x[pos[0],pos[1],0],x[pos[0],pos[1],1],x[pos[0],pos[1],2]])
       y[pos[0],pos[1]] = 0.5 * (cmin + cmax)  % lightness 0-1
    end
    lum:mat=uninit(size(image,0..1))
    parallel_do(size(lum),image,lum,getLuma_kernel)   
end

%Get Lightness using CIELab color space
%Lightness is the L term of the HSL color space. 
%This is calculated by taking the average between the minimum and maximum values of the RGB color.
function [light:mat] = getLightnessImageCIELab(image : cube)
    lab = rgb2lab(image)
    light = lab[:,:,0]/100 %Normalize between 0 and 1
end

%Luminance is the perceptual grey representation of a color. (Brightness)
%This is calculated using l: luminance as XYZ color 
function [lum:mat] = getLuminanceImage(image : cube)
    function [] = __kernel__ getLuminanceImage_kernel(x:cube'unchecked,y:mat'unchecked,pos:ivec2)
       y[pos[0],pos[1]] = 0.2126 * x[pos[0],pos[1],0] + 0.7152*x[pos[0],pos[1],1] + 0.0722*x[pos[0],pos[1],2] 
    end
    lum:mat=uninit(size(image,0..1))
    parallel_do(size(lum),image,lum,getLuminanceImage_kernel)   
end

%Get enhance bright mask, return mask and luma image
function [y:mat]=get_bright_mask(x:cube'unchecked,luma:mat'unchecked,params:object)
    %Thresholding kernel functions
    function []= __kernel__ thresholding_kernel(x:cube'unchecked,y:mat'checked,luma:mat'checked,th_luma:scalar,th_sat:scalar,pos:ivec2)
        luma_value =  luma[pos[0],pos[1]]
        %At least one channel saturated or luma > th
        if(max(x[pos[0],pos[1],0..2])>th_sat)
            y[pos[0],pos[1]] = luma_value
        elseif(luma_value > th_luma)
            y[pos[0],pos[1]] = luma_value
        endif
    end
    
    %Thresholding original image
    y=zeros(size(x,0..1)) 
    parallel_do(size(y),x,y,luma,params.th_luma,params.th_sat,thresholding_kernel) %Size is WxH    
   
    %Blur and stop 
    if(params.gf)
        y = fastguidedfilter(luma,y,200,10^-2,4);
    endif    
    y = y.^params.f
    y = (y>0).*clamp(y/max(y),1)

end

%Compute Log-average luminance
%[Tumblin and Rushmeier 1993; Ward 1994; Holm 1996]
function [k]= getLogLumaAverage(luma_image:mat'unchecked)
    ep=0.0001;
    [m,n] = size(luma_image)
    k=exp(sum(log(luma_image+ep))/(m*n));
end

%Compute Norm-Log-average luminance
%Ahmet Og˘uz Akyüz 2006
function [k]= getLogLumaNorm(luma_image:mat'unchecked)
    ep=0.0001;
    logLuma = log(luma_image+ep);
    Lm = min(luma_image)
    LM = max(luma_image)
    
    if (Lm == LM)
        d = 1
    else
        d = log(LM+ep)-log(Lm+ep)    
    endif
    [m,n] = size(luma_image)
    k=(sum(log(luma_image+ep))/(m*n)-log(Lm+ep))/d
    if(k<0.0)
        k=0.0
    endif    
end

%Apply bright mask
function [y:cube]=apply_bright_mask(x:cube'unchecked,mask:mat'unchecked, params:object)
    function []= __kernel__ linear_get_mask(x:cube'unchecked,y:cube'unchecked,mask:mat'unchecked,b:scalar,f:scalar,pos:ivec2)
        {!kernel target="gpu"}
        if(mask[pos[0],pos[1]]>0)
            %Apply Separation of Max an RGB ratio
            m=max(x[pos[0],pos[1],:]);
            ratio=x[pos[0],pos[1],:]/m;
            m = m + b*(mask[pos[0],pos[1]])
            y[pos[0],pos[1],:]=m*ratio;
        else
            y[pos[0],pos[1],:]=x[pos[0],pos[1],:]; 
        endif
    end
    %Create black image
    y=uninit(size(x))
    parallel_do(size(x,0..1),x,y,mask,params.boost_luminance,params.f,linear_get_mask) %Size is WxH
end


function [] = main()
    %Video to process
    video_file_sdr="H:\H2DR\movie_trailers\AVATAR PANDORA TRAILER HD 1080p.mp4"
    right_text_img = imread("Media/text_right.png")
    left_text_img = imread("Media/text_left.png")
    mask_text_img = imread("Media/text_mask.png")
    p=8 %8 bits original file
    n=16
    x_pos_tit = 20%150

    %%%%%%%%%%%%%% Output PNG file path %%%%%%%%%%%%
    png_out_folder = "C:/Users/ipi/Videos/"
    png_frame_counter=1
    png_out_w = 1920
    png_out_h = 1080    
                
    %%%%%%%%%%%%%%%% General Params %%%%%%%%%%%%%%%%
    general_params = object()
    peak_luminance_sim2=6000.0;
    sdr_factor = 2.5; % 600 nits
          
    %%%%%%%%%%%%%%%% Denoising PARAMS %%%%%%%%%%%%%%%
    gf_params = object()
    gf_params.r=16
    gf_params.epsf=1.2%1.12
    gf_params.eps=(gf_params.epsf)^4;
    
    %%%%%%%%%%% Automatic  dark, normal, bright classification %%%%%%%%%
    %threshold
    th_dark_norm = 0.45
    th_norm_bright = 0.65
    
    %Learning time
    l_high = 0.25
    l_low = 0.5
    
    %Values for middle gray out
    mid_out_normal = 0.18
    mid_out_dark = 0.10 %dark or dim
    mid_out_bright = 0.235
    %Values for maximum brigthness 
    max_bright_normal = 0.8
    max_bright_dark = 0.4
    max_bright_bright = 0.9
    
    %%%%%%%%%%%%  Expand operator curve %%%%%%%%%%%%%%%%
    % Default params
    eo_params = object()
    eo_params.a:scalar= 2.78 % Contrast
    eo_params.d:scalar = 0.7 % Shoulder
    eo_params.midIn:scalar=0.5
    eo_params.midOut:scalar= mid_out_normal %0.063 %This value could be change dynamically .. TODO 0.18 HDR
    eo_params.hdrMax:scalar=1.0
    eo_params.peak_luminance:scalar=0.75%max_bright_normal    %0.8  %0.67 %Peak luminance in TMO
    updateBC(eo_params);
    
    %%%%%%%%%%%% POCS  %%%%%%%%%%%%%%%
    pocs_params = object();
    pocs_params.r_pocs=3  %R?
    pocs_params.it=6
    pocs_params.steps=1
    
    %%%%%%%%%%% ENHANCE BRIGHT %%%%%%%%%%%
    eb_params = object()
    eb_params.th_luma=222/255  %Luminance Didyk
    eb_params.th_sat=230/255  %Saturation LDR2HDR
    eb_params.f=4.16 
    eb_params.gf=true; 
    eb_params.boost_luminance = 1-eo_params.peak_luminance
    eb_params.method = 0 %0-Adding the mask   %1-Multiply the mask
    
    %%%%%%%%%%%%%%%%%%%% FORMS %%%%%%%%%%%%%%%%%%%%%%%
    frm = form("Color grading")  
    frm.width = 600
    frm.height = 900
    frm.center()
    
    frm_cl = form("Dark, Normal, Bright... scene classification")  
    frm_cl.width = 500
    frm_cl.height = 150
    slider_brightness_th_luma = frm.add_slider("Luminance(th1):",eb_params.th_luma,0.1,1.1)
    frm_cl.center()
    slider_LogLumaNorm = frm_cl.add_slider("Log Luma Normalized",0,0,1.0)
    bright_status=frm_cl.add_button("Dark")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Opem stream
    stream = vidopen(video_file_sdr) % Opens the specified video file for playing

    %Variables
    s_width=stream.frame_width
    s_height=stream.frame_height
    
    frame = cube(s_height,s_width,3)
    frame_denoised = cube(s_height,s_width,3)
    frame_expanded = cube(s_height,s_width,3)
    
    frame_bright_mask = mat(s_height,s_width)
    frame_luma = mat(s_height,s_width)
   
    frame_l = cube(s_height,s_width,3)
    frame_u = cube(s_height,s_width,3)
    frame_v_buff = cube(s_height,s_width,3)
    frame_dequant = cube(s_height,s_width,3)
    
    frame_pngsave = zeros(png_out_h,png_out_w,3)
    
    frame_show = cube(s_height,s_width,3)

    sz = [s_height,s_width,3]   
    looping = false

    % Sets the frame rate for the imshow function
    sync_framerate(stream.avg_frame_rate) 
    
    % GUI
    frm.add_heading("General parameters")
    cb_moving_line = frm.add_checkbox("Moving line", false)
    cb_side_by_side = frm.add_checkbox("Compare SDR expanded", false)
    cb_sdr_vs_hdr = frm.add_checkbox("Compare HDR", false)
    cb_no_line = frm.add_checkbox("No line", true)
    cb_record = frm.add_checkbox("Recording", false )
    
    frm.add_heading("Denoising parameters (LDR non-linear space)")
    cb_denoise = frm.add_checkbox("Denoising: ", true)
    slider_denoising_epsf = frm.add_slider("Denoising factor:",gf_params.epsf,0.0,20.0)
 
    frm.add_heading("POCS")
    cb_pocs = frm.add_checkbox("POCS: ", true)
    slider_pocs_r = frm.add_slider("R:",pocs_params.r_pocs,1,20)
    slider_pocs_steps = frm.add_slider("S:",pocs_params.steps,1,20)
    
    frm.add_heading("Brightness enhancement mask computation")
    slider_brightness_th_luma = frm.add_slider("Luminance(th1):",eb_params.th_luma,0.1,1.1)
    slider_brightness_th_sat = frm.add_slider("Saturation(th2):",eb_params.th_sat,0.1,1.1)
    slider_brightness_f = frm.add_slider("Remove Glow:",eb_params.f,1.0,40)
    cb_enhance_brightness_gf = frm.add_checkbox("Edge stop and smooth: ", true)
    cb_show_brightness_mask = frm.add_checkbox("Show Mask (left) ", false)
    frm.add_heading("HDR Brightness enhancement ")
    cb_enhance_brightness_add = frm.add_checkbox("Enhance Brigthness by addition: ", true)
    cb_enhance_brightness_mult = frm.add_checkbox("Enhance Brigthness by multiplying  ", false)
    
    frm.add_heading("Color grading params")
    slider_max_lum = frm_cl.add_slider("Maximun luminance:",eo_params.peak_luminance,0.1,1)
    slider_a = frm.add_slider("Contrast(a):",eo_params.a,0.0,10.0)
    slider_d = frm.add_slider("Shoulder(d):",eo_params.d,0.0,10.0)
    %slider_midIn = frm.add_slider("Mid In :",tmo_params.midIn,0.0,max_value)
    slider_midOut =  frm_cl.add_slider("Mid Out (*) :",eo_params.midOut,0.0,1)
    
    frm.add_heading("Video Player")
    vidstate = object()
    [vidstate.is_playing, vidstate.allow_seeking, vidstate.show_next_frame] = [true, true, true]
    button_stop=frm.add_button("Stop")
    button_stop.icon = imread("Media/control_stop_blue.png")
    button_play=frm.add_button("Play")
    button_play.icon = imread("Media/control_play_blue.png")
    button_fullrewind=frm.add_button("Full rewind")
    button_fullrewind.icon = imread("Media/control_start_blue.png")
    position = frm.add_slider("Position",0,0,floor(stream.duration_sec*stream.avg_frame_rate))
    params_display = frm.add_display()
    
    %Events
    slider_max_lum.onchange.add(()-> (eo_params.peak_luminance = slider_max_lum.value;
                                      eb_params.boost_luminance = 1-slider_max_lum.value;);)
                                      
    slider_brightness_th_luma.onchange.add(()-> (eb_params.th_luma = slider_brightness_th_luma.value);)                        
    
    cb_enhance_brightness_add.onchange.add(()-> (cb_enhance_brightness_mult.value = !(cb_enhance_brightness_add.value));)                        
    
    cb_enhance_brightness_mult.onchange.add(()-> (cb_enhance_brightness_add.value = !(cb_enhance_brightness_mult.value));)                        
    
    slider_brightness_th_sat.onchange.add(()-> (eb_params.th_sat = slider_brightness_th_sat.value);)                        
    
    slider_brightness_f.onchange.add(()-> (eb_params.f = slider_brightness_f.value);)
    
    cb_enhance_brightness_gf.onchange.add(()-> (eb_params.gf = cb_enhance_brightness_gf.value);)
    
    cb_moving_line.onchange.add(()-> (cb_no_line.value=false;)) 
    
    cb_side_by_side.onchange.add(()-> (cb_no_line.value=false;
                                       cb_moving_line.value = false;))
    
    cb_sdr_vs_hdr.onchange.add(()-> (cb_no_line.value=false;
                                       cb_moving_line.value = false;))
    
    cb_no_line.onchange.add(()-> (cb_moving_line.value = false;)) 
    
    slider_a.onchange.add(()-> (eo_params.a = slider_a.value;
                                updateBC(eo_params)))      
                                
    slider_d.onchange.add(()-> (eo_params.d = slider_d.value;
                                updateBC(eo_params)))    
                                  
    slider_denoising_epsf.onchange.add(()-> (gf_params.epsf = slider_denoising_epsf.value;
                                             gf_params.eps=(gf_params.epsf)^4;))      
    
    slider_pocs_r.onchange.add(()-> (pocs_params.r_pocs = floor(slider_pocs_r.value)))                                
    
    slider_pocs_steps.onchange.add(()-> (pocs_params.steps = floor(slider_pocs_steps.value)))                                

    button_stop.onclick.add(() -> vidstate.is_playing = false)
    
    button_play.onclick.add(() -> vidstate.is_playing = true)
    
    button_fullrewind.onclick.add(() -> (vidseek(stream, 0); vidstate.show_next_frame = true))    
    
    position.onchange.add(() -> vidstate.allow_seeking ? vidseek(stream, position.value/stream.avg_frame_rate) : [])
  
    %Using mid-level mapping to adjust brightness pre-tonemappingkeeps contrast and saturation consistent 
%   slider_midIn.onchange.add(()-> (tmo_params.midIn = slider_midIn.value;
%                                updateBC(tmo_params)))          

    slider_midOut.onchange.add(()-> (eo_params.midOut = slider_midOut.value;
                                updateBC(eo_params)))                        
    %Denoising
    cb_denoise.onchange.add(()-> (denoising=cb_denoise.value;))
    
    %Curve
    val_in=0.0..1/(2^p-1)..1.0
    
    %To show the comparisson line
    xloc = floor(s_width/2)
    steps = 10
    log_luma_norm = 0.0;
    %vidseek(stream, 39)
    
    repeat
        %tic()
        if(cb_moving_line.value)
            xloc = xloc+steps
            if (xloc >= 3*s_width/4 || xloc <= s_width/4)
                steps=-steps;
            endif
        endif
        % Reads until there is no frame left. 
        if vidstate.is_playing 
            %if (!vidreadframe(stream) || !vidreadframe(stream_hdr))
            if (!vidreadframe(stream))
                if looping
                    % Jump back to the first frame
                    vidseek(stream, 0)
                else
                    break
                endif
            endif
        endif
        
        %Read frames
        frame = float(stream.rgb_data)
        
        %Denoising using fast joint bilateral filter (guided filter) 
        if(cb_denoise.value)
            frame_denoised[:,:,0] = fastguidedfilter(frame[:,:,0], frame[:,:,0], gf_params.r, gf_params.eps);
            frame_denoised[:,:,1] = fastguidedfilter(frame[:,:,1], frame[:,:,1], gf_params.r, gf_params.eps);
            frame_denoised[:,:,2] = fastguidedfilter(frame[:,:,2], frame[:,:,2], gf_params.r, gf_params.eps);
            %Clamp values
            frame_denoised=clamp(frame_denoised,255.0)
        else
            frame_denoised=frame
        endif    
        
        %Normalized
        frame_denoised = frame_denoised/255.0 %Max value

        %Get frame luminance
        frame_luma = getLuminanceImage(frame_denoised) %Not linearized
        log_luma_norm = 0.5*log_luma_norm + 0.5*getLogLumaNorm(frame_luma)
        slider_LogLumaNorm.value = log_luma_norm
        
        %Get bright mask
        frame_bright_mask = get_bright_mask(frame_denoised,frame_luma,eb_params)                                
                                                                                                
        %Calc LUT for POCS and iTMO, this lut includes linearize procedure
        lut=getLut(val_in,eo_params)
        params_display.plot(val_in,lut);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         Update mid out and maximum brightness
        reach_midOut = 0;
        reach_bright = 0;
        if(slider_LogLumaNorm.value < th_dark_norm) %Dark
            reach_midOut = mid_out_dark;
            reach_bright = max_bright_dark;
            bright_status.text = "Dark"
        elseif(slider_LogLumaNorm.value < th_norm_bright) %Normal
            reach_midOut = mid_out_normal;
            reach_bright = max_bright_normal;
            bright_status.text = "Normal"
        else %Bright > 0.8
            reach_midOut = mid_out_bright;
            reach_bright = max_bright_bright;
            bright_status.text = "Bright"
        endif
        
        % Update reach luminance
        if(reach_bright > eo_params.peak_luminance) %Up luminance
            eo_params.peak_luminance = eo_params.peak_luminance + l_high*abs(reach_bright-eo_params.peak_luminance)
            eb_params.boost_luminance = 1-eo_params.peak_luminance
            slider_max_lum.value = eo_params.peak_luminance
        elseif(reach_bright < eo_params.peak_luminance)%Down luminance
            eo_params.peak_luminance = eo_params.peak_luminance - l_low*abs(reach_bright-eo_params.peak_luminance)
            eb_params.boost_luminance = 1-eo_params.peak_luminance
            slider_max_lum.value = eo_params.peak_luminance
        endif
        
        %Update mid_out
        if(reach_midOut > eo_params.midOut) %Up luminance
            eo_params.midOut = eo_params.midOut + l_high*abs(reach_midOut-eo_params.midOut)
            slider_midOut.value = eo_params.midOut
            updateBC(eo_params)
        elseif(reach_midOut < eo_params.midOut)%Down luminancesnip
            eo_params.midOut = eo_params.midOut - l_low*abs(reach_midOut-eo_params.midOut)
            slider_midOut.value = eo_params.midOut
            updateBC(eo_params)
        endif
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %Aplly LUT and get L and H frames
        %frame_expanded = EO_LUT_CF(frame_denoised,frame_l,frame_u,eo_params,lut)
        frame_expanded = EO_LUT(frame_denoised,frame_l,frame_u,lut)
        
        %POCS
        frame_dequant = copy(frame_expanded)
        if(cb_pocs.value)
            for i = 1..6
                parallel_do(size(frame_dequant),frame_v_buff,frame_dequant,pocs_params.r_pocs,pocs_horizontal_run)
                parallel_do(size(frame_dequant),frame_dequant,frame_v_buff,pocs_params.r_pocs,frame_u,frame_l,pocs_vertical_run)
            end
        endif
        
        if(cb_enhance_brightness_add.value || cb_enhance_brightness_add.value)
            frame_dequant = apply_bright_mask(frame_dequant,frame_bright_mask,eb_params)
        endif
        
        %Create show frame
        %Show mask instead original video
        if(cb_no_line.value)
            frame_show=frame_dequant;
            %Add text
            %frame_show[x_pos_tit..x_pos_tit+43,20..20+277,:] = right_text_img[:,:,0..2]/1024;
        else
            if(cb_side_by_side.value)
                %Must be 540x960 each image
                frame_show[s_height/4..s_height/4+s_height/2-1,0..s_width/2-1,:] = imresize(linearize(frame),0.5,"nearest")/sdr_factor;
                frame_show[s_height/4..s_height/4+s_height/2-1,s_width/2..s_width-1,:] = imresize(frame_dequant,0.5,"nearest")
%            elseif(cb_sdr_vs_hdr.value)    
%                frame_show=zeros(size(frame_show)); 
%                frame_show[s_height/4..s_height/4+s_height/2-1,0..s_width/2-1,:] = imresize(linearize(frame_hdr),0.5,"nearest")/max_value;
%                frame_show[s_height/4..s_height/4+s_height/2-1,s_width/2..s_width-1,:] = imresize(frame_dequant,0.5,"nearest")
            else
                if(cb_show_brightness_mask.value)
                    frame_show[:,0..xloc,0]=frame_bright_mask[:,0..xloc]
                    frame_show[:,0..xloc,1]=frame_bright_mask[:,0..xloc]
                    frame_show[:,0..xloc,2]=frame_bright_mask[:,0..xloc]
                    %Add text
                    frame_show[x_pos_tit..x_pos_tit+43,20..20+277,:] = mask_text_img[:,:,0..2]/1024;
                else
                    frame_show[:,0..xloc,:]=linearize(frame[:,0..xloc,:]/255)
                    %frame_show[:,0..xloc,:]=frame_expanded[:,0..xloc,:]

                    %Add text
                    frame_show[x_pos_tit..x_pos_tit+43,20..20+277,:] = left_text_img[:,:,0..2]/1024;
                endif
                
                %Show processed
                frame_show[:,xloc..s_width-1,:]=frame_dequant[:,xloc..s_width-1,:]
                %Add text
                frame_show[x_pos_tit..x_pos_tit+43,1600..1600+277,:] = right_text_img[:,:,0..2]/1024;    
                %Black vertical line
                frame_show[:,xloc..xloc+1,:]=0
            endif
        endif            

    
        h = hdr_imshow(frame_show,[0,1.0])

%       %LDR       
%       im_ldr=delinearize(frame_show)*255
%       h=imshow(im_ldr,[0,255])
%       png_path = sprintf(strcat(png_out_folder,"out%08d.png"),png_frame_counter); 
%       imwrite(png_path, im_ldr)
%     
        
        % Save in PNG      
        if(cb_record.value) 
%             sim2_img = rgb2sim2(frame_show*max_value,1)
%             png_path = sprintf(strcat(png_out_folder,"out%08d.png"),png_frame_counter); 
%             imwrite(png_path, sim2_img)
             
            pause(500)
            y : cube[uint8]= h.rasterize()
            png_path = sprintf(strcat(png_out_folder,"out%08d.png"),png_frame_counter); 

            frame_pngsave[floor((1080-s_height)/2)..s_height+floor((1080-s_height)/2)-10,0..size(y,1)-1,:] = y[5..s_height-5,0..size(y,1)-1,:]
            frame_pngsave[:,s_width-26..s_width-1,:]=0;

            imwrite(png_path, frame_pngsave[size(frame_pngsave,0)-1..-1..0,:,:]) %Flip image saved
        endif

        %toc()
        %Update position slider
        position.value = stream.pts*stream.avg_frame_rate
        png_frame_counter=png_frame_counter+1
        pause(0)
    until !hold("on")
end

