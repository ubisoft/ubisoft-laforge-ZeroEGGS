#version 300 es
precision highp float;

in vec3 fragNormal;

uniform vec4 ghostColor;

out vec4 finalColor;

void main()
{
    // A little directional shading so the translucent ghost still reads as a 3D
    // pose rather than a flat blob. Alpha comes straight from ghostColor.
    vec3 n = normalize(fragNormal);
    float d = 0.55 + 0.45 * max(dot(n, normalize(vec3(0.4, 1.0, 0.4))), 0.0);
    finalColor = vec4(ghostColor.rgb * d, ghostColor.a);
}
