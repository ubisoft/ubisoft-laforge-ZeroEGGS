#version 300 es
precision highp float;

in vec2 fragTexCoord;

uniform sampler2D inputTexture;
uniform vec2 invTextureResolution;

out vec4 finalColor;

void main()
{
    const float spanMax = 4.0;
    const float reduceAmount = 1.0 / 4.0;
    const float reduceMin = (1.0 / 64.0);

    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(texture(inputTexture, fragTexCoord + (vec2(-1.0, -1.0) * invTextureResolution)).rgb, luma);
    float lumaNE = dot(texture(inputTexture, fragTexCoord + (vec2( 1.0, -1.0) * invTextureResolution)).rgb, luma);
    float lumaSW = dot(texture(inputTexture, fragTexCoord + (vec2(-1.0,  1.0) * invTextureResolution)).rgb, luma);
    float lumaSE = dot(texture(inputTexture, fragTexCoord + (vec2( 1.0,  1.0) * invTextureResolution)).rgb, luma);
    float lumaMI = dot(texture(inputTexture, fragTexCoord).rgb, luma);

    float lumaMin = min(lumaMI, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaMI, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    vec2 dir = vec2(
        -((lumaNW + lumaNE) - (lumaSW + lumaSE)),
        +((lumaNW + lumaSW) - (lumaNE + lumaSE)));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * reduceAmount), reduceMin);
    float dirRcpMin = 1.0/(min(abs(dir.x), abs(dir.y)) + dirReduce);

    dir = min(vec2(spanMax,  spanMax), max(vec2(-spanMax, -spanMax), dir * dirRcpMin)) * invTextureResolution;

    vec3 rgba0 = texture(inputTexture, fragTexCoord + dir * (1.0 / 3.0 - 0.5)).rgb;
    vec3 rgba1 = texture(inputTexture, fragTexCoord + dir * (2.0 / 3.0 - 0.5)).rgb;
    vec3 rgba2 = texture(inputTexture, fragTexCoord + dir * (0.0 / 3.0 - 0.5)).rgb;
    vec3 rgba3 = texture(inputTexture, fragTexCoord + dir * (3.0 / 3.0 - 0.5)).rgb;

    vec3 rgbA = (1.0/ 2.0) * (rgba0 + rgba1);
    vec3 rgbB = rgbA * (1.0/ 2.0) + (1.0/ 4.0) * (rgba2 + rgba3);

    float lumaB = dot(rgbB, luma);
    
    finalColor.rgb = (lumaB < lumaMin) || (lumaB > lumaMax) ? rgbA : rgbB;
}
