#version 300 es
precision highp float;

uniform float lightClipNear;
uniform float lightClipFar;

float LinearDepth(float depth, float near, float far)
{
    return (2.0 * near) / (far + near - depth * (far - near));
}

void main()
{
    gl_FragDepth = LinearDepth(gl_FragCoord.z, lightClipNear, lightClipFar);
}
