#version 300 es
precision highp float;

#define PI 3.14159265358979323846264338327950288
#define SSAO_SAMPLE_NUM 9

in vec2 fragTexCoord;

uniform sampler2D gbufferNormal;
uniform sampler2D gbufferDepth;
uniform mat4 camView;
uniform mat4 camProj;
uniform mat4 camInvProj;
uniform mat4 camInvViewProj;
uniform mat4 lightViewProj;
uniform sampler2D shadowMap;
uniform vec2 shadowInvResolution;
uniform float camClipNear;
uniform float camClipFar;
uniform float lightClipNear;
uniform float lightClipFar;
uniform vec3 lightDir;

float LinearDepth(float depth, float near, float far)
{
    return (2.0 * near) / (far + near - depth * (far - near));
}

float NonlinearDepth(float depth, float near, float far)
{
    return (((2.0 * near) / depth) - far - near) / (near - far);
}

vec3 CameraSpace(vec2 texcoord, float depth)
{
    vec4 positionClip = vec4(vec3(texcoord, NonlinearDepth(depth, camClipNear, camClipFar)) * 2.0 - 1.0, 1.0);
    vec4 position = camInvProj * positionClip;
    return position.xyz / position.w;
}

vec3 Rand(vec2 seed)
{
    return 2.0 * fract(sin(dot(seed, vec2(12.9898, 78.233))) * vec3(43758.5453, 21383.21227, 20431.20563)) - 1.0;
}

vec2 Spiral(int sampleIndex, float turns, float seed)
{
	float alpha = (float(sampleIndex) + 0.5) / float(SSAO_SAMPLE_NUM);
	float angle = alpha * (turns * 2.0 * PI) + 2.0 * PI * seed;
	return alpha * vec2(cos(angle), sin(angle));
}

out vec4 finalColor;

void main()
{
    // Compute Shadows
    
    float depth = texture(gbufferDepth, fragTexCoord).r;
    if (depth == 1.0f) { discard; }

    vec3 positionClip = vec3(fragTexCoord, NonlinearDepth(depth, camClipNear, camClipFar)) * 2.0f - 1.0f;
    vec4 fragPositionHomo = camInvViewProj * vec4(positionClip, 1);
    vec3 fragPosition = fragPositionHomo.xyz / fragPositionHomo.w;
    vec3 fragNormal = texture(gbufferNormal, fragTexCoord).xyz * 2.0 - 1.0;
    
    vec3 seed = Rand(fragTexCoord);
    
    float shadowNormalBias = 0.01;
    
    vec4 fragPosLightSpace = lightViewProj * vec4(fragPosition + shadowNormalBias * fragNormal, 1);
    fragPosLightSpace.xyz /= fragPosLightSpace.w;
    fragPosLightSpace.xyz = (fragPosLightSpace.xyz + 1.0f) / 2.0f;
    
    float shadowDepthBias = 0.000005;
    float shadowClip = float(
        fragPosLightSpace.x < +1.0 &&
        fragPosLightSpace.x >  0.0 &&
        fragPosLightSpace.y < +1.0 &&
        fragPosLightSpace.y >  0.0);
        
    float shadow = 1.0 - shadowClip * float(
        LinearDepth(fragPosLightSpace.z, lightClipNear, lightClipFar) - shadowDepthBias > 
        texture(shadowMap, fragPosLightSpace.xy + shadowInvResolution * seed.xy).r);
    
    //shadow = texture(shadowMap, fragPosLightSpace.xy).r;
    
    // Compute SSAO
    
    float bias = 0.025f;
    float radius = 0.5f;
    float turns = 7.0f;
    float intensity = 0.15f;

    vec3 norm = mat3(camView) * fragNormal;
    vec3 base = CameraSpace(fragTexCoord, texture(gbufferDepth, fragTexCoord).r);
    float occ = 0.0;
    for (int i = 0; i < SSAO_SAMPLE_NUM; i++)
    {
        vec3 next = base + radius * vec3(Spiral(i, turns, seed.x), 0.0);
        vec4 ntex = camProj * vec4(next, 1);
        vec2 sampleTexCoord = (ntex.xy / ntex.w) * 0.5f + 0.5f;
        vec3 actu = CameraSpace(sampleTexCoord, texture(gbufferDepth, sampleTexCoord).r);
        vec3 diff = actu - base;
        float vv = dot(diff, diff);
        float vn = dot(diff, norm) - bias;
        float f = max(radius*radius - vv, 0.0);
        occ += f*f*f*max(vn / (0.001 + vv), 0.0);
    }
    occ = occ / pow(radius, 6.0);

    float ssao = max(0.0, 1.0 - occ * intensity * (5.0 / float(SSAO_SAMPLE_NUM)));

    finalColor.r = ssao;
    finalColor.g = shadow;
    finalColor.b = 0.0f;
    finalColor.a = 1.0f;
}

