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
import "../../Quasar_EXR_library/src/EXR_lib/bin/Release/EXR_lib.dll"
import "C:\Users\ipi\Documents\gluzardo\eotf_pq\quasar\transfer_functions.q"
import "C:\Users\ipi\Documents\gluzardo\quasar_sim2\sim2.q"
import "C:\Users\ipi\Documents\gluzardo\experiments\itmo_curve\adaptive_itmo.q"


%Linearization and delinearization assumes that the LDR input is encoded with the sRGB transfer function
%Linearize
function y = linearize(x)
    y=sRGB_decode(x)
end

%Delinearize
function y = delinearize(x)
    y=sRGB_encode(x)
end

function y = linearizeFrame(x,p)
    %Normalize
    y=x/(2^p-1)
    y=linearize(y)
end

%Get lut for expansion.... 
%Input the 0..255 vector and the params for the expand operator
function [y:vec] = getLut(x:vec,params:object)
    %Linearize the input
    x=linearize(x)
    y=expand(x,params);
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


% 1D Horizontal filter kernel
function [] = __kernel__ pocs_horizontal_run(y : cube'unchecked, _
    x : cube' clamped'hwtex_const, r : int, pos : ivec3)
    %{!kernel_transform enable="localwindow"}
    sum = 0.0
    for m=0..2*r
        sum += x[pos + [0,m-r,0]]
    end
    y[pos] = sum/(2*r+1)
end

% 1D Vertical filter kernel
function [] = __kernel__ pocs_vertical_run(y : cube'unchecked, _
    x : cube' clamped'hwtex_const, r : int, high:cube' unchecked, low:cube'unchecked, pos : ivec3)
    %{!kernel_transform enable="localwindow"}
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


%Apply expansion on luminance channel
function [y:mat] = expandsPOCS(x:mat'unchecked,low:mat'unchecked,high:mat'unchecked,lut:vec'unchecked)
    entries = max(size(lut))
    function [] = __kernel__ expandsPOCS_kernel(x:mat'unchecked,y:mat'unchecked,low:mat'unchecked,high:mat'unchecked,lut:vec'clamped,resol:int,pos:ivec2)
        index = floor(x[pos[0],pos[1]]*(resol-1))
        y[pos[0],pos[1]] = lut[index]
        %output=expandPixel(input,a,b,c,d,p)
        low[pos[0],pos[1]] = lut[index]-(lut[index]-lut[index-1])/2
        high[pos[0],pos[1]] = lut[index]+(lut[index+1]-lut[index])/2
    end
    y = zeros(size(x))
    parallel_do(size(x),x,y,low,high,lut,entries,expandsPOCS_kernel)
end


function [] = main()
    %Video to process
    input_type=1  %0-EXR images  1-video 
    %video_file_sdr="H:/HDR_KORTFILM_PQ1K_2020_.mov"
    video_file_sdr="H:/ldr_cutoff.mov"
    % For image input
    image_frames = object()
    image_frames.start_frame = 0
    image_frames.current_frame = image_frames.start_frame
    image_frames.end_frame = 12266
    %%%%%%
   
    %% Bits
    M=16 % Bits content codification
    p=8 % 8 bits original file
    n=16 % Bits expanded
     
      
    %Images for the player    
    right_text_img = imread("Media/text_right.png")
    left_text_img = imread("Media/text_left.png")
    mask_text_img = imread("Media/text_mask.png")
    x_pos_tit = 20%150

    %%%%%%%%%%%%%% Output PNG file path %%%%%%%%%%%%
    png_out_folder = "H:\sim2_render\"
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
    gf_params.epsf=1.41%1.12
    gf_params.eps=(gf_params.epsf)^4;
    
    %%%%%%%%%%% Automatic  dark, normal, bright classification %%%%%%%%%
    %threshold
    th_dark_norm = 0.45
    th_norm_bright = 0.65
    
    %Learning time
    l_high = 0.25
    l_low = 0.5
    
    %Values for middle gray out
    mid_out_normal = 0.16
    mid_out_dark = 0.08 %dark or dim
    mid_out_bright = 0.21
    %Values for maximum brigthness 
    max_bright_normal = 0.06
    max_bright_dark = 0.4
    max_bright_bright = 0.9
    
    %%%%%%%%%%%%  Expand operator curve %%%%%%%%%%%%%%%%
    % Default params
    eo_params = object()
    eo_params.a:scalar= 2.28% Contrast
    eo_params.d:scalar = 0.96 % Shoulder
    eo_params.midIn:scalar=0.5
    eo_params.midOut:scalar= 0.33 %0.063 %This value could be change dynamically .. TODO 0.18 HDR
    eo_params.hdrMax:scalar=1.0
    eo_params.s:scalar=1.3
    eo_params.peak_luminance:scalar=peak_luminance_sim2
    updateBC(eo_params);
    
    %%%%%%%%%%%% POCS  %%%%%%%%%%%%%%%
    pocs_params = object();
    pocs_params.r_pocs=1  %R?
    pocs_params.it=7
    pocs_params.steps=1
    
    %%%%%%%%%%% ENHANCE BRIGHT %%%%%%%%%%%
    eb_params = object()
    eb_params.th_luma=222/255  %Luminance Didyk
    eb_params.th_sat=230/255  %Saturation LDR2HDR
    eb_params.f=4.16 
    eb_params.gf=true; 
    eb_params.boost_luminance = 1-eo_params.hdrMax
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
    if(input_type==0)
        stream = video_file_sdr
        %Asume that the image input are 1920x1080 UHD
        s_width=1920%stream.frame_width
        s_height=1080%stream.frame_height
    else
        stream = vidopen(video_file_sdr) % Opens the specified video file for playing
        s_width=stream.frame_width
        s_height=stream.frame_height
    endif
    
    
    frame = cube(s_height,s_width,3)
    edr = cube(s_height,s_width,3)
    frame_denoised = cube(s_height,s_width,3)
      
    frame_bright_mask = mat(s_height,s_width)
    Ld = mat(s_height,s_width)
    Lw = mat(s_height,s_width)
   
    frame_l = mat(s_height,s_width)
    frame_u = mat(s_height,s_width)
    frame_v_buff = mat(s_height,s_width)
    frame_dequant = mat(s_height,s_width)
    
    frame_pngsave : cube[uint8]= cube[uint8](s_height,s_width,3)
        
    frame_show = cube(s_height,s_width,3)

    sz = [s_height,s_width,3]   
    looping = false

    % Sets the frame rate for the imshow function
    if(input_type==1)
        sync_framerate(stream.avg_frame_rate) 
    endif    
    
    % GUI
    frm.add_heading("General parameters")
    cb_moving_line = frm.add_checkbox("Moving line", false)
    cb_side_by_side = frm.add_checkbox("Compare", false)
    %cb_sdr_vs_hdr = frm.add_checkbox("Compare HDR", false)
    cb_no_line = frm.add_checkbox("No line", true)
    cb_record = frm.add_checkbox("Recording", false )
    
    frm.add_heading("Denoising parameters (LDR non-linear space)")
    cb_denoise = frm.add_checkbox("Denoising: ", false)
    slider_denoising_epsf = frm.add_slider("Denoising factor:",gf_params.epsf,0.0,20.0)
 
    frm.add_heading("POCS")
    cb_pocs = frm.add_checkbox("POCS: ", false)
    slider_pocs_r = frm.add_slider("Low pass filter radius:",pocs_params.r_pocs,1,20)
    slider_pocs_it = frm.add_slider("Iterations:",pocs_params.steps,1,20)
    
    frm.add_heading("Brightness enhancement mask computation")
    slider_brightness_th_luma = frm.add_slider("Luminance(th1):",eb_params.th_luma,0.1,1.1)
    slider_brightness_th_sat = frm.add_slider("Saturation(th2):",eb_params.th_sat,0.1,1.1)
    slider_brightness_f = frm.add_slider("Remove Glow:",eb_params.f,1.0,40)
    cb_enhance_brightness_gf = frm.add_checkbox("Edge stop and smooth: ", true)
    cb_show_brightness_mask = frm.add_checkbox("Show Mask (left) ", false)
    frm.add_heading("HDR Brightness enhancement ")
    cb_enhance_brightness_add = frm.add_checkbox("Enhance Brigthness by addition: ", false)
    cb_enhance_brightness_mult = frm.add_checkbox("Enhance Brigthness by multiplying  ", false)
    
    frm.add_heading("Expand operator params")
    slider_max_lum = frm_cl.add_slider("Maximun luminance ratio:",eo_params.hdrMax,0.1,1)
    slider_a = frm.add_slider("Contrast(a):",eo_params.a,0.0,10.0)
    slider_d = frm.add_slider("Shoulder(d):",eo_params.d,0.0,10.0)
    slider_midIn = frm.add_slider("Mid In :",eo_params.midIn,0.0,1)
    slider_midOut =  frm_cl.add_slider("Mid Out (*) :",eo_params.midOut,0.0,1)
    slider_s = frm.add_slider("Saturation(s):",eo_params.s,1.0,10.0)
    
    frm.add_heading("Video Player")
    vidstate = object()
    [vidstate.is_playing, vidstate.allow_seeking, vidstate.show_next_frame] = [true, true, true]
    button_stop=frm.add_button("Stop")
    button_stop.icon = imread("Media/control_stop_blue.png")
    button_play=frm.add_button("Play")
    button_play.icon = imread("Media/control_play_blue.png")
    button_fullrewind=frm.add_button("Full rewind")
    button_fullrewind.icon = imread("Media/control_start_blue.png")
    if(input_type==0)
        position = frm.add_slider("Position",image_frames.current_frame,0,image_frames.end_frame)
    else
        position = frm.add_slider("Position",0,0,floor(stream.duration_sec*stream.avg_frame_rate))
    endif
    params_display = frm.add_display()
    
    %Events
    slider_max_lum.onchange.add(()-> (eo_params.hdrMax = slider_max_lum.value;
                                      eb_params.boost_luminance = 1-slider_max_lum.value;
                                      updateBC(eo_params)))
                                      
    slider_brightness_th_luma.onchange.add(()-> (eb_params.th_luma = slider_brightness_th_luma.value);)                        
    
    cb_enhance_brightness_add.onchange.add(()-> (cb_enhance_brightness_mult.value = !(cb_enhance_brightness_add.value));)                        
    
    cb_enhance_brightness_mult.onchange.add(()-> (cb_enhance_brightness_add.value = !(cb_enhance_brightness_mult.value));)                        
    
    slider_brightness_th_sat.onchange.add(()-> (eb_params.th_sat = slider_brightness_th_sat.value);)                        
    
    slider_brightness_f.onchange.add(()-> (eb_params.f = slider_brightness_f.value);)
    
    cb_enhance_brightness_gf.onchange.add(()-> (eb_params.gf = cb_enhance_brightness_gf.value);)
    
    cb_moving_line.onchange.add(()-> (cb_no_line.value=false;)) 
    
    cb_side_by_side.onchange.add(()-> (cb_no_line.value=false;
                                       cb_moving_line.value = false;))
    
%    cb_sdr_vs_hdr.onchange.add(()-> (cb_no_line.value=false;
%                                       cb_moving_line.value = false;))
%    
    cb_no_line.onchange.add(()-> (cb_moving_line.value = false;)) 
    
    slider_a.onchange.add(()-> (eo_params.a = slider_a.value;
                                updateBC(eo_params)))      
                                
    slider_d.onchange.add(()-> (eo_params.d = slider_d.value;
                                updateBC(eo_params)))    
                                  
    slider_denoising_epsf.onchange.add(()-> (gf_params.epsf = slider_denoising_epsf.value;
                                             gf_params.eps=(gf_params.epsf)^4;))      
    
    slider_pocs_r.onchange.add(()-> (pocs_params.r_pocs = floor(slider_pocs_r.value)))                                
    
    slider_pocs_it.onchange.add(()-> (pocs_params.it = floor(slider_pocs_it.value)))                                

    button_stop.onclick.add(() -> vidstate.is_playing = false)
    
    button_play.onclick.add(() -> vidstate.is_playing = true)
    if(input_type==0)
        button_fullrewind.onclick.add(() -> (image_frames.current_frame = image_frames.start_frame; vidstate.show_next_frame = true))    
        position.onchange.add(() ->  (image_frames.current_frame= floor(position.value); position.value = floor(position.value)))  
    else
        button_fullrewind.onclick.add(() -> (vidseek(stream, 0); vidstate.show_next_frame = true))
        position.onchange.add(() -> vidstate.allow_seeking ? vidseek(stream, position.value/stream.avg_frame_rate) : [])    
    endif
    
    
  
    %Using mid-level mapping to adjust brightness pre-tonemappingkeeps contrast and saturation consistent 
    slider_midIn.onchange.add(()-> (eo_params.midIn = slider_midIn.value;
                                updateBC(eo_params)))          

    slider_midOut.onchange.add(()-> (eo_params.midOut = slider_midOut.value;
                                updateBC(eo_params))) 
                                
    slider_s.onchange.add(()-> (eo_params.s = slider_s.value;
                                updateBC(eo_params)))                             
    %Denoising
    cb_denoise.onchange.add(()-> (denoising=cb_denoise.value;))
    
    %Curve
    val_in=0.0..1/(2^p-1)..1.0 %Input values p bits normalized
    
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
        if(input_type==1)
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
            frame = float(stream.rgb_data)/(2^M-1) 
                    else
            video_file = sprintf(video_file_sdr,image_frames.current_frame)
            frame = exrread(video_file).data %EXR data in [0,1]
            
            if vidstate.is_playing
                image_frames.current_frame = image_frames.current_frame + 1
            endif    
        endif
        
        %Converto to correct color space (if is neccesary)
        %frame = Rec2020TosRGB(frame) %Frame in[0,1]
        
        %clamp_values(frame,0,1)
        frame = floor(frame*(2^p-1)) %Value are now in 0,255
                
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
        
        %Linearize 
        %frame_denoised=linearizeFrame(frame_denoised,p) %Return the image in range between 0 an 1
        frame_linearized=linearizeFrame(frame_denoised,p) %Linearize frame
        
        %Get frame luminance usign the linearized image (requisite to use this function)
        Ld = getLuminanceImage(frame_linearized) 
        log_luma_norm = 0.5*log_luma_norm + 0.5*getLogLumaNorm(Ld)
        slider_LogLumaNorm.value = log_luma_norm
        
        %Get bright mask
        frame_bright_mask = get_bright_mask(frame_denoised,Ld,eb_params)                                
                                                                                                
        %Calc LUT for POCS and iTMO, this lut includes linearize procedure
        lut=expand(val_in,eo_params)% getLut(val_in,eo_params) %%--->> this step includes linearization
        g = params_display.plot(val_in,lut);
        g.title = "iTMO Curve"
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Update mid out and maximum brightness
%        reach_midOut = 0;
%        reach_bright = 0;
%        if(slider_LogLumaNorm.value < th_dark_norm) %Dark
%            reach_midOut = mid_out_dark;
%            reach_bright = max_bright_dark;
%            bright_status.text = "Dark"
%        elseif(slider_LogLumaNorm.value < th_norm_bright) %Normal
%            reach_midOut = mid_out_normal;
%            reach_bright = max_bright_normal;
%            bright_status.text = "Normal"
%        else %Bright > 0.8
%            reach_midOut = mid_out_bright;
%            reach_bright = max_bright_bright;
%            bright_status.text = "Bright"
%        endif
%        
%        
        %% Update reach luminance
%        if(reach_bright > eo_params.hdrMax) %Up luminance
%            eo_params.hdrMax = eo_params.hdrMax + l_high*abs(reach_bright-eo_params.hdrMax)
%            eb_params.boost_luminance = 1-eo_params.hdrMax
%            slider_max_lum.value = eo_params.hdrMax
%        elseif(reach_bright < eo_params.hdrMax)%Down luminance
%            eo_params.hdrMax = eo_params.hdrMax - l_low*abs(reach_bright-eo_params.hdrMax)
%            eb_params.boost_luminance = 1-eo_params.hdrMax
%            slider_max_lum.value = eo_params.hdrMax
%        endif
%        
%        %Update mid_out
%        if(reach_midOut > eo_params.midOut) %Up luminance
%            eo_params.midOut = eo_params.midOut + l_high*abs(reach_midOut-eo_params.midOut)
%            slider_midOut.value = eo_params.midOut
%            updateBC(eo_params)
%        elseif(reach_midOut < eo_params.midOut)%Down luminancesnip
%            eo_params.midOut = eo_params.midOut - l_low*abs(reach_midOut-eo_params.midOut)
%            slider_midOut.value = eo_params.midOut
%            updateBC(eo_params)
%        endif
        
              
        %Expand the dynamic range and get L and H frames
        Lw = expand(Ld,eo_params)*1000%  expandsPOCS(Ld,frame_l,frame_u,lut)
             
        %POCS
        frame_dequant = copy(Lw)
        if(cb_pocs.value)
            for i = 1..6
                parallel_do(size(frame_dequant),frame_v_buff,frame_dequant,pocs_params.r_pocs,pocs_horizontal_run)
                parallel_do(size(frame_dequant),frame_dequant,frame_v_buff,pocs_params.r_pocs,frame_u,frame_l,pocs_vertical_run)
            end
        endif
        
        if(cb_enhance_brightness_add.value || cb_enhance_brightness_add.value)
            frame_dequant = apply_bright_mask(frame_dequant,frame_bright_mask,eb_params)
        endif
        
        %Create the HDR inverse tone mapped image - EDR     
        edr =  changeLuminanceSatComp(frame_denoised,Ld,Lw,eo_params.s) %frame_dequant*peak_luminance_sim2%

        %Create show frame
        frame_show=zeros(size(frame_show));
        if(cb_no_line.value)
            frame_show=edr;
            %Add text
            %frame_show[x_pos_tit..x_pos_tit+43,20..20+277,:] = right_text_img[:,:,0..2];
        else
            if(cb_side_by_side.value)
                frame_show[s_height/4..s_height/4+s_height/2-1,0..s_width/2-1,:] = imresize(linearize(frame/255)*peak_luminance_sim2,0.5,"nearest")/sdr_factor;
                frame_show[s_height/4..s_height/4+s_height/2-1,s_width/2..s_width-1,:] = imresize(edr,0.5,"nearest")
            else
                if(cb_show_brightness_mask.value)
                    frame_show[:,0..xloc,0]=frame_bright_mask[:,0..xloc]*peak_luminance_sim2
                    frame_show[:,0..xloc,1]=frame_bright_mask[:,0..xloc]*peak_luminance_sim2
                    frame_show[:,0..xloc,2]=frame_bright_mask[:,0..xloc]*peak_luminance_sim2
                   
                    %Add text
                    frame_show[x_pos_tit..x_pos_tit+43,20..20+277,:] = mask_text_img[:,:,0..2];
                else
                    frame_show[:,0..xloc,:]=linearize(frame[:,0..xloc,:]/255)*peak_luminance_sim2
                    %Add text
                    frame_show[x_pos_tit..x_pos_tit+43,20..20+277,:] = left_text_img[:,:,0..2]/1024;
                endif
                
                %Show processed to the right
                frame_show[:,xloc..s_width-1,:]=edr[:,xloc..s_width-1,:]
                %Add text
                frame_show[x_pos_tit..x_pos_tit+43,s_width-320..s_width-20+277,:] = right_text_img[:,:,0..2]/1024;    
                %Black vertical line
                frame_show[:,xloc..xloc+1,:]=0
            endif
        endif            

        %Show the frame  
        h = hdr_imshow(frame_show,[0,6000])
        %sim2_img = rgb2sim2(frame_show,0)
        %imshow(sim2_img,[0,255])
       %LDR       
%       im_ldr=(frame_show.^0.29)
%       h=imshow(frame_show,[0,6000])
%       png_path = sprintf(strcat(png_out_folder,"out%08d.png"),png_frame_counter); 
%       imwrite(png_path, im_ldr)
%     
       
        %Save in PNG      
        if(cb_record.value) 
            pause(500)
            png_path = sprintf(strcat(png_out_folder,"out%08d.png"),png_frame_counter); 

%            sim2_img = rgb2sim2(frame_show*max_value,1)
%            png_path = sprintf(strcat(png_out_folder,"out%08d.png"),png_frame_counter); 
%            imwrite(png_path, sim2_img)

            y : cube[uint8]= h.rasterize()
            
            frame_pngsave[floor((1080-s_height)/2)..s_height+floor((1080-s_height)/2)-10,0..size(y,1)-1,:] = y[5..s_height-5,0..size(y,1)-1,:]
            frame_pngsave[:,s_width-26..s_width-1,:]=0;
            imwrite(png_path, frame_pngsave[size(frame_pngsave,0)-1..-1..0,:,:]) %Flip image saved
        endif

        %Update position slider
        if(input_type==0)
            position.value = image_frames.current_frame
        else
            position.value = stream.pts*stream.avg_frame_rate
        endif
        png_frame_counter=png_frame_counter+1
        pause(0)
    until !hold("on")
end

