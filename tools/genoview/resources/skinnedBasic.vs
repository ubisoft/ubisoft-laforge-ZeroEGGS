#version 300 es

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;
in vec4 vertexBoneIds;
in vec4 vertexBoneWeights;

#define MAX_BONE_NUM 128
uniform mat4 boneMatrices[MAX_BONE_NUM];

uniform mat4 mvp;

out vec3 fragPosition;
out vec2 fragTexCoord;
out vec4 fragColor;
out vec3 fragNormal;

void main()
{
    int boneIndex0 = int(vertexBoneIds.x);
    int boneIndex1 = int(vertexBoneIds.y);
    int boneIndex2 = int(vertexBoneIds.z);
    int boneIndex3 = int(vertexBoneIds.w);
    
    vec4 skinnedPosition =
        vertexBoneWeights.x * (boneMatrices[boneIndex0] * vec4(vertexPosition, 1.0f)) +
        vertexBoneWeights.y * (boneMatrices[boneIndex1] * vec4(vertexPosition, 1.0f)) + 
        vertexBoneWeights.z * (boneMatrices[boneIndex2] * vec4(vertexPosition, 1.0f)) + 
        vertexBoneWeights.w * (boneMatrices[boneIndex3] * vec4(vertexPosition, 1.0f));
    
    vec3 skinnedNormal = normalize(
        vertexBoneWeights.x * (mat3(boneMatrices[boneIndex0]) * vertexNormal) +
        vertexBoneWeights.y * (mat3(boneMatrices[boneIndex1]) * vertexNormal) +
        vertexBoneWeights.z * (mat3(boneMatrices[boneIndex2]) * vertexNormal) +
        vertexBoneWeights.w * (mat3(boneMatrices[boneIndex3]) * vertexNormal));

    fragPosition = skinnedPosition.xyz / skinnedPosition.w;
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = skinnedNormal;

    gl_Position = mvp * vec4(fragPosition, 1.0f);
}
