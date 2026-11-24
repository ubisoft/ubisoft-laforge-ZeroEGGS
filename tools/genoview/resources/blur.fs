#version 300 es
precision highp float;

in vec2 fragTexCoord;

uniform sampler2D gbufferNormal;
uniform sampler2D gbufferDepth;
uniform sampler2D inputTexture;
uniform mat4 camInvProj;
uniform float camClipNear;
uniform float camClipFar;
uniform vec2 invTextureResolution;
uniform vec2 blurDirection;

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

float FastNegExp(float x)
{
    return 1.0f / (1.0f + x + 0.48f*x*x + 0.235f*x*x*x);
}

out vec4 finalColor;

void main()
{
    float depth = texture(gbufferDepth, fragTexCoord).r;
    if (depth == 1.0f) { discard; }

    vec3 baseNormal = texture(gbufferNormal, fragTexCoord).rgb * 2.0f - 1.0f;
    vec3 basePosition = CameraSpace(fragTexCoord, depth);
    
    vec4 totalColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    float totalWeight = 0.0f;
    float stride = 2.0f;
    
    for (int x = -3; x <= 3; x++)
    {
        vec2 sampleTexcoord = fragTexCoord + float(x) * stride * blurDirection * invTextureResolution;
        vec4 sampleColour = texture(inputTexture, sampleTexcoord);
        vec3 sampleNormal = texture(gbufferNormal, sampleTexcoord).rgb * 2.0f - 1.0f;
        vec3 samplePosition = CameraSpace(sampleTexcoord, texture(gbufferDepth, sampleTexcoord).r);
        
        vec3 diffPosition = (samplePosition - basePosition) / 0.05f;

        float weightPosition = FastNegExp(dot(diffPosition, diffPosition));
        float weightNormal = max(dot(sampleNormal, baseNormal), 0.0f);

        float weight = weightPosition * weightNormal;

        totalColor += weight * sampleColour;
        totalWeight += weight;
    }
    
    finalColor = totalColor / totalWeight;    
    
    //finalColor = texture(inputTexture, fragTexCoord);
}
