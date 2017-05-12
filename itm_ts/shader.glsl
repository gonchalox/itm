#version 120
#extension GL_ARB_texture_rectangle : enable

uniform sampler2DRect texInput;
uniform int interpolationMode;
uniform int colorMap;
uniform vec4 colorFilter;
uniform vec4 colorMapVals[8];
uniform float gamma;
uniform vec3 coordOffset;
uniform vec3 coordScale;


#define RGB2XYZ_XR			0.412137
#define RGB2XYZ_XG			0.376845
#define RGB2XYZ_XB			0.142235
#define RGB2XYZ_YR			0.221852
#define RGB2XYZ_YG			0.757303
#define RGB2XYZ_YB			0.081677
#define RGB2XYZ_ZR			0.011588
#define RGB2XYZ_ZG			0.069829
#define RGB2XYZ_ZB			0.706036
//Defined 1/08/2016
#define LUMINANCE_BOOST			6000.0000

vec3 RGBtoXYZ( vec3 RGB )
{
	float X = RGB2XYZ_XR * RGB.r + RGB2XYZ_XG * RGB.g + RGB2XYZ_XB * RGB.b;
	float Y = RGB2XYZ_YR * RGB.r + RGB2XYZ_YG * RGB.g + RGB2XYZ_YB * RGB.b;
	float Z = RGB2XYZ_ZR * RGB.r + RGB2XYZ_ZG * RGB.g + RGB2XYZ_ZB * RGB.b;
	return vec3(X, Y, Z);
}

float XYZtou( float X, float Y, float Z)
{
	return (((1626.6875 * X) / (X + 15.0 * Y + 3.0 * Z)) + 0.546875) * 4.0;
}

float XYZtov( float X, float Y, float Z)
{
	return (((3660.046875 * Y) / (X + 15.0 * Y + 3.0 * Z)) + 0.546875) * 4.0;
}

vec3 RGBtoXYZforuv( vec3 RGB )
{
	return RGBtoXYZ(RGB);
	
}

float RGBtou( vec3 RGB )
{
	vec3 XYZ = RGBtoXYZforuv(RGB);
	return XYZtou( XYZ.r, XYZ.g, XYZ.b);
}

float RGBtov( vec3 RGB )
{
	vec3 XYZ = RGBtoXYZforuv(RGB);
	return XYZtov( XYZ.r, XYZ.g, XYZ.b);
}


void main() {
	vec4 pixelVal;

	vec2 texCoordOut = coordOffset.st + coordScale.st * gl_TexCoord[0].st;

	pixelVal = texture2DRect(texInput, gl_TexCoord[0].st);

	// Step 2. affine intensity transform
	pixelVal = colorFilter.a + colorFilter * pixelVal;

	// Step 3. gamma correction
	if (gamma != 1.0)
	{
		pixelVal = pow(pixelVal, vec4(gamma,gamma,gamma,1.0));
	}

	// Step 4. color maps
	if (colorMap > 0)
	{
		float grayVal = dot(pixelVal, vec4(0.2126, 0.7152, 0.0722, 0.0)) * (colorMap-1);
		grayVal = clamp(grayVal, 0.0f, float(colorMap-1));
		int ind = int(floor(grayVal));
		float fr = grayVal - ind;
		pixelVal = colorMapVals[ind] * (1.0f - fr) + colorMapVals[ind + 1] * fr;
	}


	vec3 In_RGB = max(pixelVal.rgb,vec3(0.0,0.0,0.0))*LUMINANCE_BOOST;
	vec3 Out_RGB;
	vec3 XYZ;
	float Y;
	float L, u, v;
	float Lh, Ll, Ch, Cl;
	float alpha = 0.0376;			
	float Lscale = 32;				
	int odd = mod(texCoordOut.s, 2) >= 1 ? 1 : 0;

	XYZ = RGBtoXYZ(In_RGB);
	Y = XYZ.g;
    u = RGBtou(In_RGB);
	v = RGBtov(In_RGB);

	//if (Y < 0.00001)
	if (Y < 0.0000152587890625)
	{
		L = 0.0000152587890625;
	}
	else
	{
		L = (alpha * log2(Y)) + 0.5;
	}
	L = (253 * L + 1) * Lscale;
	L = floor(L + 0.5);
	L = clamp( L, 32, 8159);		
	Lh = floor(L / 32.0);			
	Ll = L - (Lh * 32.0);	
		
	if (odd == 0)
	{
		v = floor(v + .5);
		v = clamp( v, 4, 1019);		
		Ch = floor(v / 4.0);		
		Cl = v - (Ch * 4.0);		

		Out_RGB.r = (Ll * 8.0) + (Cl * 2.0);
		Out_RGB.r = clamp(Out_RGB.r, 1, 254);	
		Out_RGB.g = Lh;
		Out_RGB.b = Ch;
	}
	else
	{
		u = floor(u + .5);
		u = clamp( u, 4, 1019);		
		Ch = floor(u / 4.0);		
		Cl = u - (Ch * 4.0);		

		Out_RGB.r = Ch;
		Out_RGB.g = Lh;
		Out_RGB.b = (Ll * 8.0) + (Cl * 2.0);
		Out_RGB.b = clamp(Out_RGB.b, 1, 254);
	}

	Out_RGB /= 255.0;

	gl_FragColor = vec4(clamp(Out_RGB,0.0,1.0), 1.0);
}
