import pygame
from OpenGL.GL import *
from ctypes import sizeof, c_void_p
 
DEFAULT_VERTEX_SHADER = '''
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;
 
    out vec2 TexCoords;
 
    void main()
    {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
'''
 
DEFAULT_FRAGMENT_SHADER = '''
    #version 330 core
    out vec4 FragColor;
 
    in vec2 TexCoords;
 
    uniform sampler2D texture1;
 
    void main()
    {
      FragColor = texture(texture1, TexCoords);
    }
'''
 
default_shader = None
def get_default_shader():
    global default_shader
 
    if default_shader:
        return default_shader
 
    def _create_shader(shader_type, source):
        """compile a shader"""
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(shader))
 
        return shader
 
    vertex   = _create_shader(GL_VERTEX_SHADER, DEFAULT_VERTEX_SHADER)
    fragment = _create_shader(GL_FRAGMENT_SHADER, DEFAULT_FRAGMENT_SHADER)
 
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex)
    glAttachShader(shader_program, fragment)
    glBindFragDataLocation(shader_program, 0, "outColor")
    glLinkProgram(shader_program)
 
    default_shader = shader_program
    glUseProgram(shader_program)
    glUniform1i(glGetUniformLocation(shader_program, 'texture1'), 0)
 
 
    return default_shader
 
class Text:
    def __init__(
        self,
        text,
        position   = (0.0, 0.0),
        font_name  = 'dejavusans',
        font_size  = 60,
        font_color = (1.0, 1.0, 0.0, 1.0),
        bg_color   = None
    ):
        self.text       = text
        self.x, self.y  = position
        self.font_name  = font_name
        self.font_size  = font_size
        self.font_color = font_color
        self.bg_color   = bg_color
 
        self.texture = glGenTextures(1)
        self.vbo = glGenBuffers(1)
        self.vao = glGenVertexArrays(1)
        self.ebo = glGenBuffers(1)
 
        self._setup()
 
    def _pygameize_color(self, color):
        return None if color == None or color[3] == 0.0 else [i * 255 for i in color]
 
    def _setup(self):
        font = pygame.font.SysFont(self.font_name, self.font_size)
        surface = font.render(self.text, True, self._pygameize_color(self.font_color), self._pygameize_color(self.bg_color))
        # copy surface data to openGL texture
        self.surface_to_texture(surface, self.texture)
 
        # calc vertex positions
        xres, yres = pygame.display.get_surface().get_size()
        width, height = surface.get_size()
        x1 = width / xres / 2
        y1 = height / yres / 2
 
        vertices = [
            self.x + x1, self.y + y1, 1.0, 1.0, # top right
            self.x + x1, self.y - y1, 1.0, 0.0, # bottom right
            self.x - x1, self.y - y1, 0.0, 0.0, # bottom left
            self.x - x1, self.y + y1, 0.0, 1.0, # top left
        ]
        vertices = (GLfloat * len(vertices))(*vertices) # cast to GLfloat
 
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW)
 
        elements = [
            0, 1, 3,
            1, 2, 3
        ]
        elements = (GLuint * len(elements))(*elements)
 
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_DYNAMIC_DRAW)
 
        glBindVertexArray(self.vao)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), c_void_p(2 * sizeof(GLfloat)))
 
    def draw(self, shader):
        glUseProgram(shader)
 
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
 
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
 
        glBindVertexArray(0)
 
    def set_text(self, text):
        self.text = text
        self._setup()
        return self
 
    def surface_to_texture(self, surface, texture=None, wrap_s=GL_REPEAT, wrap_t=GL_REPEAT, min_filter=GL_LINEAR, mag_filter=GL_LINEAR):
        texture = texture if texture else glGenTextures(1)
        width, height = surface.get_size()
        data = pygame.image.tostring(surface, 'RGBA', True)
        glBindTexture(GL_TEXTURE_2D, texture)
        # textures - wrapping
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t)
        # textures - filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glGenerateMipmap(GL_TEXTURE_2D)
 
        return texture
