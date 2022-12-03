import json
import os.path
from math import *
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def vec_length(vec):
    return sqrt(vec[0]**2+vec[1]**2+vec[2]**2)

def vec_norm(vec):
    length = vec_length(vec)
    if length == 0:
        return (0,0,0)
    norm = ( vec[0]/length , vec[1]/length , vec[2]/length )
    return norm

def cross_product(v1,v2):
    return (v1[1]*v2[2]-v1[2]*v2[1] , v1[2]*v2[0]-v1[0]*v2[2] , v1[0]*v2[1]-v1[1]*v2[0] )

def dot_product(v1, v2):
    ret = 0.0

    for i in range(len(v1)):
        ret += v1[i]*v2[i]

    return ret

ANGLE=pi/1960
SCREEN_SIZE = (1024,800)
VECTORS = [(10,32),(20,47)]
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
#GREEN = (0, 255, 0)
GREEN = (0.0, 1.0, 0.0, .5)
LIGHT_GRAY=(220,220,220)

SCREEN_VEC1 = (1,0,0)
SCREEN_VEC2 = (0,1,0)
SCREEN_NORMAL = vec_norm(cross_product(SCREEN_VEC1, SCREEN_VEC2))
SHOW_INDICES = True
SHOW_NORMALS = True

SCALE = 1.0
CENTER = (SCREEN_SIZE[0]//2,SCREEN_SIZE[1]//2)
NOTE = 5

def kor_down(angle):
    return(cos(angle),sin(angle,0))

def kor_up(angle):
    return(cos(angle),sin(angle,2))



def prism(n,d1=2,d2=2,height=2):
    coords = []
    for i in range(n):
        angle = (2*pi/n)*i;
        x_low=cos(angle)*d1
        y_low=sin(angle)*d1
        coords.append( (x_low,y_low,-height/2) )
    for i in range(n):
        angle = (2*pi/n)*i;
        x_high=cos(angle)*d2
        y_high=sin(angle)*d2
        coords.append( (x_high,y_high,height/2) )

    faces = []
    for j in range(n):
        if j == (n-1):
            faces += [ [j,0, n, 2*n-1] ]
        else:
            faces += [ [j,j+1,j+n+1,j+n] ]
    faces += [ list(range(n-1,-1,-1)) ]
    faces += [ list(range(n,2*n)) ]

    return coords,faces

def f1(x):
    return x**2

def parabola_2d(n,z):
    l = 0
    r = 6
    points = []
    for i in range(l*(n-1),r*(n-1)+1,r-l):
        x = cos(i/(n-1))
        y = sin(i/(n-1))
        points.append((x,y,z))
    return points

def parabola_3d(n,h):
    vertices = []
    vertices.extend(parabola_2d(n,h/2))
    vertices.extend(parabola_2d(n,-h/2))
    faces = []
    face1=[]
    for i in range(n):
        face1.append(i)
    faces.append(list(reversed(face1)))
    face2=[]
    for j in range(n, 2*n):
        face2.append(j)
    faces.append(face2)
    for j in range(n):
        if j == (n-1):
            faces += [ [j,0, n, 2*n-1] ]
            #continue
        else:
            faces += [ [j,j+1,j+n+1,j+n] ]
    return vertices, faces






MODELS = [
    (*parabola_3d(5,1), "Parabola")
]






CURRENT_MODEL = 0
VERTICES, FACES, NAME = MODELS[CURRENT_MODEL]

def next_model(direction):
    global VERTICES,FACES,CURRENT_MODEL, NAME
    CURRENT_MODEL = (CURRENT_MODEL + direction) % len(MODELS)
    VERTICES, FACES, NAME = MODELS[CURRENT_MODEL]

def normal(face):
    VERTICES=MODELS[CURRENT_MODEL][0]
    ver1=VERTICES[face[0]]
    ver2=VERTICES[face[1]]
    ver3=VERTICES[face[2]]
    vec1=(ver2[0]-ver1[0],ver2[1]-ver1[1],ver2[2]-ver1[2])
    vec2=(ver3[0]-ver2[0],ver3[1]-ver2[1],ver3[2]-ver2[2])
    return vec_norm(cross_product(vec1, vec2))

def is_visible_face(face):
    norm = normal(face)
    if dot_product(SCREEN_NORMAL, norm)<=0:
        return True
    else:
        return False


def filter_faces():
    visible= []
    invisible = []
    for face in FACES:
        if is_visible_face(face):
            visible += [face]
        else:
            invisible += [face]
    return visible, invisible

def matrix_rotate_x(angle):
    MX = [
        [1,0,0],
        [0,cos(angle),-sin(angle)],
        [0,sin(angle),cos(angle)]
    ]
    return MX

def matrix_rotate_y(angle):
    MY = [
        [cos(angle),0,sin(angle)],
        [0,1,0],
        [-sin(angle),0,cos(angle)]
    ]
    return MY

def matrix_rotate_z(angle):
    MZ = [
        [cos(angle),-sin(angle),0],
        [sin(angle),cos(angle),0],
        [0,0,1]
    ]
    return MZ

def rotate(angle, rot):
    m = rot(angle)
    for i in range(len(VERTICES)):
        VERTICES[i]=product(m, VERTICES[i])




def product(m,v):
    """ Matrix product """
    result = []
    for row in m:
        ret=0
        for i in range(0,len(row)):
            xm = row[i]
            xv = v[i]
            ret += xv*xm
        result.append(ret)
    return result


def to_scr(pos):
    x=pos[0]*SCALE + CENTER[0]
    y=pos[1]*SCALE + CENTER[1]
    return(x,y)


def orthogonal(pos):
    return (pos[0],pos[1])


def draw_wireframe_face(screen, color, face):
    for i in range(len(face)):
        a = VERTICES[face[i]]
        if (i+1) == len(face):
            b = VERTICES[face[0]]
        else:
            b = VERTICES[face[i+1]]
        ortho_a = orthogonal(a)
        screen_ortho_a = to_scr(ortho_a)
        ortho_b = orthogonal(b)
        screen_ortho_b = to_scr(ortho_b)
        pygame.draw.line(screen, color, screen_ortho_a, screen_ortho_b ,5)

def draw_shaded_face(screen, color, face):
    points = []
    for i in range(len(face)):
        points.append(to_scr(orthogonal(VERTICES[face[i]])))
    very_new_color = turn_to_color(color , face)
    pygame.draw.polygon(screen, very_new_color, points)
    return

def draw_wire_GL():
    mode = GL_LINE_LOOP
    mode = GL_POLYGON
    coords, faces, _ = MODELS[CURRENT_MODEL]
    glMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE, GREEN)
    glColor3d(0,1,0)
    for face in faces[:]:
        glBegin(mode)
        for idx in reversed(face):
            glVertex3fv(coords[idx])
        glEnd()


def turn_to_color(base_color, face):
    face_normal = normal(face)
    dot_face_normal = -dot_product(face_normal, SCREEN_NORMAL)
    new_color = (weighted_average(base_color[0]/2,base_color[0], dot_face_normal) , weighted_average(base_color[1]/2,base_color[1],dot_face_normal) , weighted_average(base_color[2]/2,base_color[2], dot_face_normal))
    return new_color

def weighted_average(left,right,alpha):
    return int(alpha*left+(1-alpha)*right)

def draw_shaded(screen, color, faces):
    for face in faces:
        draw_shaded_face(screen, color, face)


def draw_wireframe(screen, color, faces):
    for face in faces:
        draw_wireframe_face(screen, color, face)



def draw_vec_center(screen, color, vec):
    x=vec[0]
    y=vec[1]
    pygame.draw.line(screen, color, CENTER, (SCREEN_SIZE[0]//2+x,SCREEN_SIZE[1]//2-y),3)
    return

def draw_vert_index(screen, font):
    if not SHOW_INDICES:
        return
    for i in range(len(VERTICES)):
        x = to_scr(orthogonal(VERTICES[i]))
        pic = font.render("%d"%i, False, BLACK)
        screen.blit(pic, x)

def draw_normals(screen):
    if not SHOW_NORMALS:
        return
    for face in FACES:
        norm = normal(face)
        mid = VERTICES[face[0]]
        norm = (mid[0]+norm[0],mid[1]+norm[1], mid[2]+norm[2])
        pygame.draw.line(screen, BLACK, to_scr(mid), to_scr(norm), 2)
    return


def main():
    global SCALE, NOTE, SHOW_INDICES, SHOW_NORMALS
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((1024,800),DOUBLEBUF|OPENGL)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (1024/800), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glTranslatef(0.0, 0.0, -5)
    font = pygame.font.SysFont('Sans', 32)
    title_font = pygame.font.SysFont('Sans', 48, bold=True)


    glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 2, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0, 0, 0, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
    glEnable(GL_DEPTH_TEST)

    glScalef(SCALE, SCALE, SCALE)
    run = True
    x_rotation = False
    y_rotation = False
    z_rotation = False
    while run:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                run=False
            elif ev.type == pygame.KEYDOWN:
                sign = 1
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    sign = -1

                if ev.key==pygame.K_UP:
                    VECTORS[0]=(VECTORS[0][0],VECTORS[0][1]+10)
                elif ev.key == pygame.K_q:
                    run = False
                elif ev.key == pygame.K_x:
                    x_rotation = True
                elif ev.key == pygame.K_z:
                    z_rotation = True
                elif ev.key == pygame.K_y:
                    y_rotation = True
                elif ev.key == pygame.K_a:
                    if(NOTE > 3 or sign > 0):
                        NOTE = NOTE + 1 * (sign)
                        MODELS[0] = (*parabola_3d(NOTE, 2), "Parabola %s"%NOTE)
                        next_model(0)
                        print(NOTE)
                    else:
                        pass#run = False
                elif ev.key == pygame.K_i:
                    SHOW_INDICES = not SHOW_INDICES
                elif ev.key == pygame.K_n:
                    SHOW_NORMALS = not SHOW_NORMALS
                elif ev.key == pygame.K_m:
                    next_model(sign)
            elif ev.type == pygame.KEYUP:
                if ev.key == pygame.K_x:
                    x_rotation = False
                elif ev.key == pygame.K_z:
                    z_rotation = False
                elif ev.key == pygame.K_y:
                    y_rotation = False
                elif ev.key == pygame.K_RIGHT:
                    SCALE = 2.
                    glScalef(SCALE, SCALE, SCALE)
                elif ev.key == pygame.K_LEFT:
                    SCALE = 0.5
                    glScalef(SCALE, SCALE, SCALE)

        if not run:
            break
        print(SCALE)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        if x_rotation:
            glRotatef(sign*ANGLE*360, 1,0,0)
        if y_rotation:
            glRotatef(sign * ANGLE*360, 0, 1, 0)
        if z_rotation:
            glRotatef(sign * ANGLE*360, 0, 0, 1)

        draw_wire_GL()
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)
        pygame.display.flip()
    pygame.quit()




if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print(error)
        raise
    pygame.quit()
