clear;
clc;

TOT_IMGS = 1000;

OUTPUT_FOLDER='imgs/';

IMAGE_SIZE=[600, 600, 3];

SHAPE_SIZE=[75, 150]; %min/max

BORDER_SIZE=[5, 25]; %min/max

INTENSITY_BACKGROUND=[-30, 30]; %min/max

TOT_INTENSITY_LEVELS=20;

if ~exist(OUTPUT_FOLDER,'dir')
    mkdir(OUTPUT_FOLDER);
end

probs_shape=[0.25, 0.25, 0.25, 0.25];   %circle, triangle, rectangle, losangle
probs_shape=cumsum(probs_shape);


for i=1:TOT_IMGS
    img=uint8(randi(255,IMAGE_SIZE))+randi(diff(INTENSITY_BACKGROUND))+INTENSITY_BACKGROUND(1);
    img(img<0)=0;
    img(img>255)=255;
    intensity=round((randi(TOT_INTENSITY_LEVELS)-1)*255/(TOT_INTENSITY_LEVELS-1));
    border_color=randi(3);
    border_size=BORDER_SIZE(1)+randi(BORDER_SIZE(2));
    
    %shape
    idx=find(rand()<=probs_shape);
    if idx(1)==1
        rad=SHAPE_SIZE(1)+randi(SHAPE_SIZE(2));
        cx=randi(IMAGE_SIZE(2)-2*rad)+rad;
        cy=randi(IMAGE_SIZE(1)-2*rad)+rad;
                        
        [xx, yy] = meshgrid(1:IMAGE_SIZE(2),1:IMAGE_SIZE(1));        
        for c=1:3
            ch=img(:,:,c);
            ch(((xx-cx).^2+(yy-cy).^2)<rad^2)=intensity;
            if border_color==c
                ch((((xx-cx).^2+(yy-cy).^2)<rad^2)&(((xx-cx).^2+(yy-cy).^2)>=(rad-border_size)^2))=255;
            else
                ch((((xx-cx).^2+(yy-cy).^2)<rad^2)&(((xx-cx).^2+(yy-cy).^2)>=(rad-border_size)^2))=0;
            end
                
            img(:,:,c)=ch;
        end
                        
    end
    if idx(1)==2
        s=SHAPE_SIZE(1)+randi(SHAPE_SIZE(2));
        cx=randi(IMAGE_SIZE(2)-2*s)+s;
        cy=randi(IMAGE_SIZE(1)-2*s)+s;                
        mask = poly2mask([cx-s cx cx+s], [cy+s cy-s cy+s], size(img,1), size(img,2));              
        mask_border=logical(mask-imerode(mask,strel('disk',border_size)));
                                
        for c=1:3
            ch=img(:,:,c);
            ch(mask)=intensity;
            if border_color==c
                ch(mask_border)=255;
            else
                ch(mask_border)=0;
            end
            img(:,:,c)=ch;
        end
    end        
    if idx(1)==3
        sx=SHAPE_SIZE(1)+randi(SHAPE_SIZE(2));
        sy=SHAPE_SIZE(1)+randi(SHAPE_SIZE(2));
        cx=randi(IMAGE_SIZE(2)-2*sx)+sx;
        cy=randi(IMAGE_SIZE(1)-2*sy)+sy;
        
        for c=1:3
            ch=img(:,:,c);
            ch(cy:cy+sy,cx:cx+sx)=intensity;  
            if border_color==c
                ch(cy:cy+border_size,cx:cx+sx)=255;
                ch(cy+sy-border_size:cy+sy,cx:cx+sx)=255;                
                ch(cy:cy+sy,cx:cx+border_size)=255;
                ch(cy:cy+sy,cx+sx-border_size:cx+sx)=255;
            else
                ch(cy:cy+border_size,cx:cx+sx)=0;
                ch(cy+sy-border_size:cy+sy,cx:cx+sx)=0;
                ch(cy:cy+sy,cx:cx+border_size)=0;
                ch(cy:cy+sy,cx+sx-border_size:cx+sx)=0;
            end
            img(:,:,c)=ch;
        end
    end
    
    if idx(1)==4
        s=SHAPE_SIZE(1)+randi(SHAPE_SIZE(2));
        cx=randi(IMAGE_SIZE(2)-2*s)+s;
        cy=randi(IMAGE_SIZE(1)-2*s)+s;                
        mask = poly2mask([cx-s cx cx+s cx], [cy cy-s cy cy+s], size(img,1), size(img,2));    
        mask_border=logical(mask-imerode(mask,strel('disk',border_size)));
        for c=1:3
            ch=img(:,:,c);
            ch(mask)=intensity;  
            if border_color==c
                ch(mask_border)=255;
            else
                ch(mask_border)=0;
            end
            img(:,:,c)=ch;
        end
        
    end
        
        
    
    imwrite(imrotate(img, (randi(4)-1)*90,'crop'),[OUTPUT_FOLDER, 'img_',num2str(i),'.jpg']);
end

