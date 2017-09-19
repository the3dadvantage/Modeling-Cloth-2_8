# You are at the top. If you attempt to go any higher
#   you will go beyond the known limits of the code
#   universe where there are most certainly monsters

# to do:
#  use point raycaster to make a cloth_wrap option
#  add sewing springs
#  set up better indexing so that edges only get calculated once
#  self colisions
#  object collisions
#  add bending springs
#  add curl by shortening bending springs on one axis or diagonal
#  independantly scale bending springs and structural to create buckling
#  run on frame update as an option?
#  option to cache animation?
#  collisions need to properly exclude pinned and vertex pinned
#  virtual springs do something wierd to the velocity
#  multiple converging springs go to far. Need to divide by number of springs at a vert or move them all towards a mean

# now!!!!
#  refresh self collisions.

# collisions:
# Onlny need to check on of the edges for groups connected to a vertex    
# for edge to face intersections...
# figure out where the edge hit the face
# figure out which end of the edge is inside the face
# move along the face normal to the surface for the point inside.
# if I reflect by flipping the vel around the face normal
#   if it collides on the bounce it will get caught on the next iteration


'''??? Would it make sense to do self collisions with virtual edges ???'''

bl_info = {
    "name": "Modeling Cloth",
    "author": "Rich Colburn, email: the3dadvantage@gmail.com",
    "version": (1, 0),
    "blender": (2, 78, 0),
    "location": "View3D > Extended Tools > Modeling Cloth",
    "description": "Maintains the surface area of an object so it behaves like cloth",
    "warning": "There might be an angry rhinoceros behind you",
    "wiki_url": "",
    "category": '3D View'}


import bpy
import bmesh
import numpy as np
from numpy import newaxis as nax
from bpy_extras import view3d_utils
from ctypes import windll, Structure, c_ulong, byref
import time


class POINT(Structure):
    _fields_ = [("x", c_ulong), ("y", c_ulong)]


def queryMousePosition():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return { "x": pt.x, "y": pt.y}


you_have_a_sense_of_humor = False
#you_have_a_sense_of_humor = True
if you_have_a_sense_of_humor:
    import antigravity


def get_last_object():
    """Finds cloth objects for keeping settings active
    while selecting other objects like pins"""
    cloths = [i for i in bpy.data.objects if i.modeling_cloth] # so we can select an empty and keep the settings menu up
    if bpy.context.object.modeling_cloth:
        return cloths, bpy.context.object
    
    if len(cloths) > 0:
        ob = extra_data['last_object']
        return cloths, ob
    return None, None


def get_poly_centers(ob, type=np.float32):
    mod = False
    m_count = len(ob.modifiers)
    if m_count > 0:
        show = np.zeros(m_count, dtype=np.bool)
        ren_set = np.copy(show)
        ob.modifiers.foreach_get('show_render', show)
        ob.modifiers.foreach_set('show_render', ren_set)
        mod = True
    mesh = ob.to_mesh(bpy.context.scene, True, 'RENDER')
    p_count = len(mesh.polygons)
    center = np.zeros(p_count * 3)#, dtype=type)
    mesh.polygons.foreach_get('center', center)
    center.shape = (p_count, 3)
    bpy.data.meshes.remove(mesh)
    if mod:
        ob.modifiers.foreach_set('show_render', show)

    return center


def get_poly_normals(ob, type=np.float32):
    mod = False
    m_count = len(ob.modifiers)
    if m_count > 0:
        show = np.zeros(m_count, dtype=np.bool)
        ren_set = np.copy(show)
        ob.modifiers.foreach_get('show_render', show)
        ob.modifiers.foreach_set('show_render', ren_set)
        mod = True
    mesh = ob.to_mesh(bpy.context.scene, True, 'RENDER')
    p_count = len(mesh.polygons)
    normal = np.zeros(p_count * 3)#, dtype=type)
    mesh.polygons.foreach_get('normal', normal)
    normal.shape = (p_count, 3)
    bpy.data.meshes.remove(mesh)
    if mod:
        ob.modifiers.foreach_set('show_render', show)

    return normal


def get_v_normals(ob, type=np.float32):
    mod = False
    m_count = len(ob.modifiers)
    if m_count > 0:
        show = np.zeros(m_count, dtype=np.bool)
        ren_set = np.copy(show)
        ob.modifiers.foreach_get('show_render', show)
        ob.modifiers.foreach_set('show_render', ren_set)
        mod = True
    mesh = ob.to_mesh(bpy.context.scene, True, 'RENDER')
    v_count = len(mesh.vertices)
    normal = np.zeros(v_count * 3)#, dtype=type)
    mesh.vertices.foreach_get('normal', normal)
    normal.shape = (v_count, 3)
    bpy.data.meshes.remove(mesh)
    if mod:
        ob.modifiers.foreach_set('show_render', show)

    return normal


def closest_point_edge(e1, e2, p):
    '''Returns the location of the point on the edge'''
    vec1 = e2 - e1
    vec2 = p - e1
    d = np.dot(vec2, vec1) / np.dot(vec1, vec1)
    cp = e1 + vec1 * d 
    return cp


def generate_collision_data(ob, pins, means):
    """The mean of each face is treated as a point then
    checked against the cpoe of the normals of every face.
    If the distance between the cpoe and the center of the face
    it's checking is in the margin, do a second check to see
    if it's also in the virtual cylinder around the normal.
    If it's in the cylinder, back up along the normal until the
    point is on the surface of the cylinder.
    instead of backing up along the normal, could do a line-plane intersect
    on the path of the point against the normal. 
    
    Since I know the direction the points are moving... it might be possible
    to identify the back sides of the cylinders so the margin on the back
    side is infinite. This way I could never miss the collision.
    !!! infinite backsides !!! Instead... since that would work because
    there is no way to tell the difference between a point that's moving
    away from the surface and a point that's already crossed the surface...
    I could do a raycast onto the infinte plane of the normal still treating
    it like a cylinder so instead of all the inside triangle stuff, just check
    the distance from the intersection to the cpoe of the normal
    
    Get cylinder sizes by taking the center of each face,
    and measuring the distance to it's closest neighbor, then back up a bit
    
    !!!could save the squared distance around the cylinders and along the normal 
    to save a step while checking... !!!
    """
    
    # one issue: oddly shaped triangles can cause gaps where face centers could pass through
    # since both sids are being checked it's less likely that both sides line up and pass through the gap
    obm = bmesh.new()
    obm.from_mesh(ob.data)

    obm.faces.ensure_lookup_table()
    obm.verts.ensure_lookup_table()
    p_count = len(obm.faces)
    
    per_face_v =  [[v.co for v in f.verts] for f in obm.faces]
    
    # $$$$$ calculate means with add.at like it's set up already$$$$$
    #means = np.array([np.mean([v.co for v in f.verts], axis=0) for f in obm.faces], dtype=np.float32)
    ### !!! calculating means from below wich is dynamic. Might be better since faces change size anyway. Could get better collisions

    # get sqared distance to closest vert in each face. (this will still work if the mesh isn't flat)
    sq_dist = []
    for i in range(p_count):
        dif = np.array(per_face_v[i]) - means[i]
        sq_dist.append(np.min(np.einsum('ij,ij->i', dif, dif)))
    
    # neighbors for excluding point face collisions.
    neighbors = np.tile(np.ones(p_count, dtype=np.bool), (pins.shape[0], 1))
    #neighbors = np.tile(pins, (pins.shape[0], 1))
    p_neighbors = [[f.index for f in obm.verts[p].link_faces] for p in pins]
    for x in range(neighbors.shape[0]):
        neighbors[x][p_neighbors[x]] = False

    # returns the radius distance from the mean to the closest vert in the polygon
    return np.array(np.sqrt(sq_dist), dtype=np.float32), neighbors
        

def create_vertex_groups(groups=['common', 'not_used'], weights=[0.0, 0.0], ob=None):
    '''Creates vertex groups and sets weights. "groups" is a list of strings
    for the names of the groups. "weights" is a list of weights corresponding 
    to the strings. Each vertex is assigned a weight for each vertex group to
    avoid calling vertex weights that are not assigned. If the groups are
    already present, the previous weights will be preserved. To reset weights
    delete the created groups'''
    if ob is None:
        ob = bpy.context.object
    vg = ob.vertex_groups
    for g in range(0, len(groups)):
        if groups[g] not in vg.keys(): # Don't create groups if there are already there
            vg.new(groups[g])
            vg[groups[g]].add(range(0,len(ob.data.vertices)), weights[g], 'REPLACE')
        else:
            vg[groups[g]].add(range(0,len(ob.data.vertices)), 0, 'ADD') # This way we avoid resetting the weights for existing groups.


def get_bmesh(obj=None):
    ob = get_last_object()[1]
    if ob is None:
        ob = obj
    obm = bmesh.new()
    if ob.mode == 'OBJECT':
        obm.from_mesh(ob.data)
    elif ob.mode == 'EDIT':
        obm = bmesh.from_edit_mesh(ob.data)
    return obm


def get_minimal_edges(ob):
    obm = get_bmesh(ob)
    obm.edges.ensure_lookup_table()
    obm.verts.ensure_lookup_table()
    obm.faces.ensure_lookup_table()
    
    # get sew edges:
    sew = [i.index for i in obm.edges if len(i.link_faces)==0]
    
    # get linear edges
    e_count = len(obm.edges)
    eidx = np.zeros(e_count * 2, dtype=np.int32)
    e_bool = np.zeros(e_count, dtype=np.bool)
    e_bool[sew] = True
    ob.data.edges.foreach_get('vertices', eidx)
    eidx.shape = (e_count, 2)

    # get diagonal edges:
    diag_eidx = []
    print('new============')
    start = 0
    stop = 0
    step_size = [len(i.verts) for i in obm.faces]
    p_v_count = np.sum(step_size)
    p_verts = np.ones(p_v_count, dtype=np.int32)
    ob.data.polygons.foreach_get('vertices', p_verts)
    # can only be understood on a good day when the coffee flows (uses rolling and slicing)
    # creates uniqe diagonal edge sets
    for f in obm.faces:
        fv_count = len(f.verts)
        stop += fv_count
        if fv_count > 3: # triangles are already connected by linear springs
            skip = 2
            f_verts = p_verts[start:stop]
            for fv in range(len(f_verts)):
                if fv > 1:        # as we go around the loop of verts in face we start overlapping
                    skip = fv + 1 # this lets us skip the overlap so we done have mirror duplicates
                roller = np.roll(f_verts, fv)
                for r in roller[skip:-1]:
                    diag_eidx.append([roller[0], r])

        start += fv_count    
    
    # eidx groups
    sew_eidx = eidx[e_bool]
    lin_eidx = eidx[-e_bool]
    diag_eidx = np.array(diag_eidx)
        
    return lin_eidx, diag_eidx, sew_eidx


def add_virtual_springs(remove=False):
    cloth = data[get_last_object()[1].name]
    obm = get_bmesh()
    obm.verts.ensure_lookup_table()
    count = len(obm.verts)
    idxer = np.arange(count)
    sel = np.array([v.select for v in obm.verts])    
    selected = idxer[sel]

    if remove:
        ls = cloth.virtual_springs[:, 0]
        
        in_sel = np.in1d(ls, idxer[sel])

        deleter = np.arange(ls.shape[0])[in_sel]
        reduce = np.delete(cloth.virtual_springs, deleter, axis=0)
        cloth.virtual_springs = reduce
        
        if cloth.virtual_springs.shape[0] == 0:
            cloth.virtual_springs.shape = (0, 2)

        return

    existing = np.append(cloth.eidx, cloth.virtual_springs, axis=0)
    ls = existing[:,0]
    springs = []
    for i in idxer[sel]:

        # to avoid duplicates:
        # where this vert occurs on the left side of the existing spring list
        v_in = existing[i == ls]
        v_in_r = v_in[:,1]
        not_in = selected[-np.in1d(selected, v_in_r)]
        idx_set = not_in[not_in != i]
        for sv in idx_set:
            springs.append([i, sv])
    virtual_springs = np.array(springs, dtype=np.int32)
    
    if virtual_springs.shape[0] == 0:
        virtual_springs.shape = (0, 2)
    
    cloth.virtual_springs = np.append(cloth.virtual_springs, virtual_springs, axis=0)
    # gets appended to eidx in the cloth_init function after calling get connected polys in case geometry changes


def generate_guide_mesh():
    """Makes the icosphere that appears when creating pins"""
    verts = [[0.0, 0.0, 0.0], [-0.01, -0.01, 0.1], [-0.01, 0.01, 0.1], [0.01, -0.01, 0.1], [0.01, 0.01, 0.1], [-0.03, -0.03, 0.1], [-0.03, 0.03, 0.1], [0.03, 0.03, 0.1], [0.03, -0.03, 0.1], [-0.01, -0.01, 0.2], [-0.01, 0.01, 0.2], [0.01, -0.01, 0.2], [0.01, 0.01, 0.2]]
    edges = [[0, 5], [5, 6], [6, 7], [7, 8], [8, 5], [1, 2], [2, 4], [4, 3], [3, 1], [5, 1], [2, 6], [4, 7], [3, 8], [9, 10], [10, 12], [12, 11], [11, 9], [3, 11], [9, 1], [2, 10], [12, 4], [6, 0], [7, 0], [8, 0]]
    faces = [[0, 5, 6], [0, 6, 7], [0, 7, 8], [0, 8, 5], [1, 3, 11, 9], [1, 2, 6, 5], [2, 4, 7, 6], [4, 3, 8, 7], [3, 1, 5, 8], [12, 10, 9, 11], [4, 2, 10, 12], [3, 4, 12, 11], [2, 1, 9, 10]]
    name = 'ModelingClothPinGuide'
    if 'ModelingClothPinGuide' in bpy.data.objects:
        mesh_ob = bpy.data.objects['ModelingClothPinGuide']
    else:   
        mesh = bpy.data.meshes.new('ModelingClothPinGuide')
        mesh.from_pydata(verts, edges, faces)  
        mesh.update()
        mesh_ob = bpy.data.objects.new(name, mesh)
        bpy.context.scene.objects.link(mesh_ob)
        mesh_ob.show_x_ray = True
    return mesh_ob


def create_giude():
    """Spawns the guide"""
    if 'ModelingClothPinGuide' in bpy.data.objects:
        mesh_ob = bpy.data.objects['ModelingClothPinGuide']
        return mesh_ob
    mesh_ob = generate_guide_mesh()
    bpy.context.scene.objects.active = mesh_ob
    bpy.ops.object.material_slot_add()
    if 'ModelingClothPinGuide' in bpy.data.materials:
        mat = bpy.data.materials['ModelingClothPinGuide']
    else:    
        mat = bpy.data.materials.new(name='ModelingClothPinGuide')
    mat.use_transparency = True
    mat.alpha = 0.35            
    mat.emit = 2     
    mat.game_settings.alpha_blend = 'ALPHA_ANTIALIASING'
    mat.diffuse_color = (1, 1, 0)
    mesh_ob.material_slots[0].material = mat
    return mesh_ob


def delete_giude():
    """Deletes the icosphere"""
    if 'ModelingClothPinGuide' in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects['ModelingClothPinGuide'])
    if 'ModelingClothPinGuide' in bpy.data.meshes:        
        guide_mesh = bpy.data.meshes['ModelingClothPinGuide']
        guide_mesh.user_clear()
        bpy.data.meshes.remove(guide_mesh)
    

def scale_source(multiplier):
    """grow or shrink the source shape"""
    ob = get_last_object()[1]
    if ob is not None:
        if ob.modeling_cloth:
            count = len(ob.data.vertices)
            co = np.zeros(count*3, dtype=np.float32)
            ob.data.shape_keys.key_blocks['modeling cloth source key'].data.foreach_get('co', co)
            co.shape = (count, 3)
            mean = np.mean(co, axis=0)
            co -= mean
            co *= multiplier
            co += mean
            ob.data.shape_keys.key_blocks['modeling cloth source key'].data.foreach_set('co', co.ravel())                
            if hasattr(data[ob.name], 'cy_dists'):
                data[ob.name].cy_dists *= multiplier
            

def reset_shapes():
    """Sets the modeling cloth key to match the source key.
    Will regenerate shape keys if they are missing"""
    if bpy.context.object.modeling_cloth:
        ob = bpy.context.object
    else:    
        ob = extra_data['last_object']

    if ob.data.shape_keys == None:
        ob.shape_key_add('Basis')    
    if 'modeling cloth source key' not in ob.data.shape_keys.key_blocks:
        ob.shape_key_add('modeling cloth source key')        
    if 'modeling cloth key' not in ob.data.shape_keys.key_blocks:
        ob.shape_key_add('modeling cloth key')        
        ob.data.shape_keys.key_blocks['modeling cloth key'].value=1
    
    keys = ob.data.shape_keys.key_blocks
    count = len(ob.data.vertices)
    co = np.zeros(count * 3, dtype=np.float32)
    keys['modeling cloth source key'].data.foreach_get('co', co)
    keys['modeling cloth key'].data.foreach_set('co', co)

    data[ob.name].vel *= 0
    
    ob.data.shape_keys.key_blocks['modeling cloth key'].mute = True
    ob.data.shape_keys.key_blocks['modeling cloth key'].mute = False


def update_pin_group():
    """Updates the cloth data after changing mesh or vertex weight pins"""
    create_instance(new=False)


def collision_data_update(self, context):
    if self.modeling_cloth_self_collision:    
        create_instance(new=False)    


def refresh_noise(self, context):
    if self.name in data:
        zeros = np.zeros(data[self.name].count, dtype=np.float32)
        random = np.random.random(data[self.name].count)
        zeros[:] = random
        data[self.name].noise = ((zeros + -0.5) * self.modeling_cloth_noise * 0.1)[:, nax]


class Cloth(object):
    pass


def create_instance(new=True):
    if new:    
        cloth = Cloth()
        cloth.ob = bpy.context.object
        
        cloth.pin_list = []
        cloth.hook_list = []
        cloth.virtual_springs = np.empty((0,2), dtype=np.int32)
        cloth.sew_springs = []
    
    else:
        ob = get_last_object()[1]
        cloth = data[ob.name]
        cloth.ob = ob 
    bpy.context.scene.objects.active = cloth.ob
    mode = cloth.ob.mode
    if mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
        
    cloth.name = cloth.ob.name
    if cloth.ob.data.shape_keys == None:
        cloth.ob.shape_key_add('Basis')    
    if 'modeling cloth source key' not in cloth.ob.data.shape_keys.key_blocks:
        cloth.ob.shape_key_add('modeling cloth source key')        
    if 'modeling cloth key' not in cloth.ob.data.shape_keys.key_blocks:
        cloth.ob.shape_key_add('modeling cloth key')        
        cloth.ob.data.shape_keys.key_blocks['modeling cloth key'].value=1
    cloth.count = len(cloth.ob.data.vertices)

    if 'modeling_cloth_pin' not in cloth.ob.vertex_groups:
        cloth.pin_group = create_vertex_groups(groups=['modeling_cloth_pin'], weights=[0.0], ob=None)
    for i in range(cloth.count):
        try:
            cloth.ob.vertex_groups['modeling_cloth_pin'].weight(i)
        except RuntimeError:
            # assign a weight of zero
            cloth.ob.vertex_groups['modeling_cloth_pin'].add(range(0,len(cloth.ob.data.vertices)), 0.0, 'REPLACE')
    cloth.pin_bool = -np.array([cloth.ob.vertex_groups['modeling_cloth_pin'].weight(i) for i in range(cloth.count)], dtype=np.bool)

    # unique edges------------>>>
    uni_edges = get_minimal_edges(cloth.ob)
    if len(uni_edges[1]) > 0:   
        cloth.eidx = np.append(uni_edges[0], uni_edges[1], axis=0)
    else:
        cloth.eidx = uni_edges[0]
    #cloth.eidx = uni_edges[0][0]

    if cloth.virtual_springs.shape[0] > 0:
        cloth.eidx = np.append(cloth.eidx, cloth.virtual_springs, axis=0)
    cloth.eidx_tiler = cloth.eidx.T.ravel()    

    eidx1 = np.copy(cloth.eidx)
    pindexer = np.arange(cloth.count, dtype=np.int32)[cloth.pin_bool]
    unpinned = np.in1d(cloth.eidx_tiler, pindexer)
    cloth.eidx_tiler = cloth.eidx_tiler[unpinned]    
    cloth.unpinned = unpinned

    cloth.sew_edges = uni_edges[2]
    
    # unique edges------------>>>
    
    cloth.pcount = pindexer.shape[0]
    cloth.pindexer = pindexer
    
    cloth.sco = np.zeros(cloth.count * 3, dtype=np.float32)
    cloth.ob.data.shape_keys.key_blocks['modeling cloth source key'].data.foreach_get('co', cloth.sco)
    cloth.sco.shape = (cloth.count, 3)
    cloth.co = np.zeros(cloth.count * 3, dtype=np.float32)
    cloth.ob.data.shape_keys.key_blocks['modeling cloth key'].data.foreach_get('co', cloth.co)
    cloth.co.shape = (cloth.count, 3)    
    co = cloth.co
    cloth.vel = np.zeros(cloth.count * 3, dtype=np.float32)
    cloth.vel_start = np.zeros(cloth.count * 3, dtype=np.float32)
    cloth.vel_start.shape = (cloth.count, 3)
    cloth.vel.shape = (cloth.count, 3)
    
    #noise---
    noise_zeros = np.zeros(cloth.count, dtype=np.float32)
    random = np.random.random(cloth.count)
    noise_zeros[:] = random
    cloth.noise = ((noise_zeros + -0.5) * cloth.ob.modeling_cloth_noise * 0.1)[:, nax]
    
    cloth.waiting = False
    cloth.clicked = False # for the grab tool
    
    uni = np.unique(cloth.eidx_tiler, return_inverse=True, return_counts=True)

    cloth.mix = (1/uni[2][uni[1]])[:, nax].astype(np.float32) # force gets divided by number of springs
    cloth.mix = cloth.mix
    
    self_col = cloth.ob.modeling_cloth_self_collision
    if self_col:
        # collision======:
        # collision======:
        cloth.p_count = len(cloth.ob.data.polygons)
        cloth.p_means = get_poly_centers(cloth.ob)
        
        # could put in a check in case int 32 isn't big enough...
        cloth.cy_dists, cloth.point_mean_neighbors = generate_collision_data(cloth.ob, pindexer, cloth.p_means)
        cloth.cy_dists *= cloth.ob.modeling_cloth_self_collision_cy_size
        
        nei = cloth.point_mean_neighbors.ravel() # eliminate neighbors for point in face check
        print(np.arange(cloth.count).shape)
        print(pindexer.shape)
        cloth.v_repeater = np.repeat(pindexer, cloth.p_count)[nei]
        cloth.p_repeater = np.tile(np.arange(cloth.p_count, dtype=np.int32),(cloth.count,))[nei]
        cloth.bool_repeater = np.ones(cloth.p_repeater.shape[0], dtype=np.bool)
        
        cloth.mean_idxer = np.arange(cloth.p_count)
        cloth.mean_tidxer = np.tile(cloth.mean_idxer, (cloth.count, 1))
        
        # collision======:
        # collision======:

    
    bpy.ops.object.mode_set(mode=mode)
    return cloth


def run_handler(cloth):
    if not cloth.ob.modeling_cloth_pause:
        if cloth.ob.mode == 'EDIT':
            cloth.waiting = True
        if cloth.waiting:    
            if cloth.ob.mode == 'OBJECT':
                update_pin_group()

        if not cloth.waiting:
    
            eidx = cloth.eidx # world's most important variable

            cloth.ob.data.shape_keys.key_blocks['modeling cloth source key'].data.foreach_get('co', cloth.sco.ravel())
            sco = cloth.sco
            sco.shape = (cloth.count, 3)
            cloth.ob.data.shape_keys.key_blocks['modeling cloth key'].data.foreach_get('co', cloth.co.ravel())
            co = cloth.co
            co.shape = (cloth.count, 3)

            svecs = sco[eidx[:, 1]] - sco[eidx[:, 0]]
            sdots = np.einsum('ij,ij->i', svecs, svecs)

            co[cloth.pindexer] += cloth.noise[cloth.pindexer]
            cloth.noise *= cloth.ob.modeling_cloth_noise_decay

            cloth.vel_start[:] = co
            force = cloth.ob.modeling_cloth_spring_force
            mix = cloth.mix * force
            for x in range(cloth.ob.modeling_cloth_iterations):    

                vecs = co[eidx[:, 1]] - co[eidx[:, 0]]
                dots = np.einsum('ij,ij->i', vecs, vecs)
                div = np.nan_to_num(sdots / dots)
                swap = vecs * np.sqrt(div)[:, nax]
                move = (vecs - swap)
                tiled_move = np.append(move, -move, axis=0)[cloth.unpinned] * mix # * mix for stability: force multiplied by 1/number of springs
                
                np.add.at(cloth.co, cloth.eidx_tiler, tiled_move)
                
                if len(cloth.pin_list) > 0:
                    hook_co = np.array([cloth.ob.matrix_world.inverted() * i.matrix_world.to_translation() for i in cloth.hook_list])
                    cloth.co[cloth.pin_list] = hook_co

            if cloth.ob.modeling_cloth_sew != 0:
                if len(cloth.sew_edges) > 0:
                    sew_edges = cloth.sew_edges
                    rs = co[sew_edges[:,1]]
                    ls = co[sew_edges[:,0]]
                    sew_vecs = (rs - ls) * 0.5 * cloth.ob.modeling_cloth_sew
                    co[sew_edges[:,1]] -= sew_vecs
                    co[sew_edges[:,0]] += sew_vecs
                    



            #target a mesh======================:
            #target a mesh======================:
                
                #use the v_normal drop or possible a new closest point on mesh method
                #   as a force to move the cloth towards another object using a distance off
                #   the mesh as a target.
            
            #target a mesh======================:
            #target a mesh======================:        


            #collision=====================================
            #collision=====================================
            # for grow and shrink, the distance will need to change
            #   it gets recaluclated when going in and out of edit mode already...
            # calc velocity
            vel_dif = cloth.vel_start - cloth.co            
            cloth.vel += vel_dif


            self_col = cloth.ob.modeling_cloth_self_collision

            if self_col:
                V3 = [] # because I'm multiplying the vel by this value and it doesn't exist unless there are collisions
                col_margin = cloth.ob.modeling_cloth_self_collision_margin
                sq_margin = col_margin ** 2
                
                cloth.p_means = get_poly_centers(cloth.ob)
                
                #======== collision tree---
                # start with the greatest dimension(if it's flat on the z axis, it will return everything so start with an axis with the greatest dimensions)
                order = np.argsort(cloth.ob.dimensions) # last on first since it goes from smallest to largest
                axis_1 = cloth.co[:, order[2]]
                axis_2 = cloth.co[:, order[1]]
                axis_3 = cloth.co[:, order[0]]
                center_1 = cloth.p_means[:, order[2]]
                center_2 = cloth.p_means[:, order[1]]
                center_3 = cloth.p_means[:, order[0]]
                
                V = cloth.v_repeater # one set of verts for each face
                P = cloth.p_repeater # faces repeated in order to aling to v_repearter
                
                check_1 = np.abs(axis_1[V] - center_1[P]) < cloth.cy_dists[P]
                V1 = V[check_1]
                P1 = P[check_1]
                C1 = cloth.cy_dists[P1]
                
                check_2 = np.abs(axis_2[V1] - center_2[P1]) < C1
                
                V2 = V1[check_2]
                P2 = P1[check_2]
                C2 = C1[P2]            

                check_3 = np.abs(axis_3[V2] - center_3[P2]) < C2

                v_hits = V2[check_3]
                p_hits = P2[check_3]
                #======== collision tree end ---
                if p_hits.shape[0] > 0:        
                    # now do closest point edge with points on normals
                    normals = get_poly_normals(cloth.ob)[p_hits]
                    
                    base_vecs = cloth.co[v_hits] - cloth.p_means[p_hits]
                    d = np.einsum('ij,ij->i', base_vecs, normals) / np.einsum('ij,ij->i', normals, normals)        
                    cp = normals * d[:, nax]
                    
                    # now measure the distance along the normal to see if it's in the cylinder
                    
                    cp_dot = np.einsum('ij,ij->i', cp, cp)
                    in_margin = cp_dot < sq_margin
                    
                    if in_margin.shape[0] > 0:
                        V3 = v_hits[in_margin]
                        #P3 = p_hits[in_margin]
                        cp3 = cp[in_margin]
                        cpd3 = cp_dot[in_margin]
                        
                        d1 = sq_margin
                        d2 = cpd3
                        div = d1/d2
                        surface = cp3 * np.sqrt(div)[:, nax]

                        force = np.nan_to_num(surface - cp3)
                        force *= cloth.ob.modeling_cloth_self_collision_force
                        
                        cloth.co[V3] += force

                        cloth.vel[V3] *= .2                   
                        #np.add.at(cloth.co, V3, force * .5)
                        #np.multiply.at(cloth.vel, V3, 0.2)
                        
                        # could get some speed help by iterating over a dict maybe
                        #if False:    
                        #if True:    
                            #for i in range(len(P3)):
                                #cloth.co[cloth.v_per_p[P3[i]]] -= force[i]
                                #cloth.vel[cloth.v_per_p[P3[i]]] -= force[i]
                                #cloth.vel[cloth.v_per_p[P3[i]]] *= 0.2
                                #cloth.vel[cloth.v_per_p[P3[i]]] += cloth.vel[V3[i]]
                                #need to transfer the velocity back and forth between hit faces.

            #collision=====================================


            
            
            # floor ---
            if cloth.ob.modeling_cloth_floor:    
                floored = cloth.co[:,2] < 0        
                cloth.vel[:,2][floored] *= -1
                cloth.vel[floored] *= .1
                cloth.co[:, 2][floored] = 0
            # floor ---            
            



            
            # inflate
            inflate = cloth.ob.modeling_cloth_inflate * .1
            if inflate != 0:
                v_normals = get_v_normals(cloth.ob)
                v_normals *= inflate
                cloth.vel -= v_normals            
            
            cloth.vel[:,2][cloth.pindexer] -= cloth.ob.modeling_cloth_gravity * .01
            cloth.vel *= cloth.ob.modeling_cloth_velocity
            co[cloth.pindexer] -= cloth.vel[cloth.pindexer]        

            if len(cloth.pin_list) > 0:
                cloth.co[cloth.pin_list] = hook_co
                cloth.vel[cloth.pin_list] = 0
            
            if cloth.clicked: # for the grab tool
                for v in range(len(extra_data['vidx'])):   
                    loc = extra_data['stored_vidx'][v] + extra_data['move']
                    cloth.co[extra_data['vidx'][v]] = loc            
                    cloth.vel[extra_data['vidx'][v]] *= 0

            cloth.ob.data.shape_keys.key_blocks['modeling cloth key'].data.foreach_set('co', cloth.co.ravel())
            cloth.ob.data.shape_keys.key_blocks['modeling cloth key'].mute = True
            cloth.ob.data.shape_keys.key_blocks['modeling cloth key'].mute = False

def modeling_cloth_handler(scene):
    items = data.items()
    if len(items) == 0:
        for i in bpy.data.objects:
            i.modeling_cloth = False
        
        for i in bpy.app.handlers.scene_update_post:
            if i.__name__ == 'modeling_cloth_handler':
                bpy.app.handlers.scene_update_post.remove(i)
    
    for i, cloth in items:    
        if i in bpy.data.objects:
            run_handler(cloth)
        else:
            del(data[i])
            break

def pause_update(self, context):
    if not self.modeling_cloth_pause:
        update_pin_group()


def init_cloth(self, context):
    global data, extra_data
    data = bpy.context.scene.modeling_cloth_data_set
    extra_data = bpy.context.scene.modeling_cloth_data_set_extra
    extra_data['alert'] = False
    extra_data['drag_alert'] = False
    extra_data['last_object'] = self
    extra_data['clicked'] = False
    
    # iterate through dict: for i, j in d.items()
    if self.modeling_cloth:
        cloth = create_instance() # generate an instance of the class
        data[cloth.name] = cloth  # store class in dictionary using the object name as a key
    
    cull = [] # can't delete dict items while iterating
    for i, value in data.items():
        if not value.ob.modeling_cloth:
            cull.append(i) # store keys to delete
    
    for i in cull:
        del data[i]
    
    # could keep the handler unless there are no modeling cloth objects active
    if modeling_cloth_handler in bpy.app.handlers.scene_update_post:
        bpy.app.handlers.scene_update_post.remove(modeling_cloth_handler)
    
    if len(data) > 0:
        bpy.app.handlers.scene_update_post.append(modeling_cloth_handler)


def main(context, event):
    """Raycaster for placing pins"""
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for reg in area.regions:
                if reg.type == 'WINDOW':
                    region = reg
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    if hasattr(space, 'region_3d'):
                        rv3d = space.region_3d
    
    user32 = windll.user32
    screensize = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
    
    X= region.x
    Y= region.y
    top = screensize[1]

    win_x = bpy.context.window_manager.windows[0].x
    win_y = bpy.context.window_manager.windows[0].y

    flipped = top - (event['y'] + Y + win_y)
    
    coord = (event['x'] - win_x - X, flipped)

    view3d_utils.region_2d_to_location_3d
    
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    ray_target = ray_origin + view_vector
    
    guide = create_giude()

    def visible_objects_and_duplis():
        """Loop over (object, matrix) pairs (mesh only)"""

        for obj in context.visible_objects:
            if obj.type == 'MESH':
                if obj.modeling_cloth:    
                    yield (obj, obj.matrix_world.copy())

    def obj_ray_cast(obj, matrix):
        """Wrapper for ray casting that moves the ray into object space"""

        # get the ray relative to the object
        matrix_inv = matrix.inverted()
        ray_origin_obj = matrix_inv * ray_origin
        ray_target_obj = matrix_inv * ray_target
        ray_direction_obj = ray_target_obj - ray_origin_obj

        # cast the ray
        success, location, normal, face_index = obj.ray_cast(ray_origin_obj, ray_direction_obj)

        if success:
            return location, normal, face_index
        else:
            return None, None, None

    # cast rays and find the closest object
    best_length_squared = -1.0
    best_obj = None
    for obj, matrix in visible_objects_and_duplis():
        hit, normal, face_index = obj_ray_cast(obj, matrix)
        if hit is not None:
            hit_world = matrix * hit
            vidx = [v for v in obj.data.polygons[face_index].vertices]
            verts = np.array([matrix * obj.data.shape_keys.key_blocks['modeling cloth key'].data[v].co for v in obj.data.polygons[face_index].vertices])
            vecs = verts - np.array(hit_world)
            closest = vidx[np.argmin(np.einsum('ij,ij->i', vecs, vecs))]
            length_squared = (hit_world - ray_origin).length_squared
            if best_obj is None or length_squared < best_length_squared:
                best_length_squared = length_squared
                best_obj = obj
                guide.location = matrix * obj.data.shape_keys.key_blocks['modeling cloth key'].data[closest].co
                extra_data['latest_hit'] = matrix * obj.data.shape_keys.key_blocks['modeling cloth key'].data[closest].co
                extra_data['name'] = obj.name
                extra_data['obj'] = obj
                extra_data['closest'] = closest
                
                if extra_data['just_clicked']:
                    extra_data['just_clicked'] = False
                    best_length_squared = length_squared
                    best_obj = obj
                   

class ModelingClothPin(bpy.types.Operator):
    """Modal ray cast for placing pins"""
    bl_idname = "view3d.modeling_cloth_pin"
    bl_label = "Modeling Cloth Pin"
    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self):
        bpy.ops.object.select_all(action='DESELECT')    
        extra_data['just_clicked'] = False
        
    def modal(self, context, event):
        bpy.context.window.cursor_set("CROSSHAIR")
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'NUMPAD_0',
        'NUMPAD_PERIOD','NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4',
         'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9'}:
            # allow navigation
            return {'PASS_THROUGH'}
        elif event.type == 'MOUSEMOVE':
            pos = queryMousePosition()
            main(context, pos)
            return {'RUNNING_MODAL'}
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if extra_data['latest_hit'] is not None:
                if extra_data['name'] is not None:
                    closest = extra_data['closest']
                    name = extra_data['name']
                    e = bpy.data.objects.new('modeling_cloth_pin', None)
                    bpy.context.scene.objects.link(e)
                    e.location = extra_data['latest_hit']
                    e.show_x_ray = True
                    e.select = True
                    e.empty_draw_size = .1
                    data[name].pin_list.append(closest)
                    data[name].hook_list.append(e)
                    extra_data['latest_hit'] = None
                    extra_data['name'] = None        
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            delete_giude()
            cloths = [i for i in bpy.data.objects if i.modeling_cloth] # so we can select an empty and keep the settings menu up
            extra_data['alert'] = False
            if len(cloths) > 0:                                        #
                ob = extra_data['last_object']                         #
                bpy.context.scene.objects.active = ob
            bpy.context.window.cursor_set("DEFAULT")
            return {'CANCELLED'}

            
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        cloth_objects = False
        cloths = [i for i in bpy.data.objects if i.modeling_cloth] # so we can select an empty and keep the settings menu up
        if len(cloths) > 0:
            cloth_objects = True        
            extra_data['alert'] = True
            
        if context.space_data.type == 'VIEW_3D' and cloth_objects:
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            extra_data['alert'] = False
            bpy.context.window.cursor_set("DEFAULT")
            return {'CANCELLED'}


# drag===================================
# drag===================================
#[‘DEFAULT’, ‘NONE’, ‘WAIT’, ‘CROSSHAIR’, ‘MOVE_X’, ‘MOVE_Y’, ‘KNIFE’, ‘TEXT’, ‘PAINT_BRUSH’, ‘HAND’, ‘SCROLL_X’, ‘SCROLL_Y’, ‘SCROLL_XY’, ‘EYEDROPPER’]

def main_drag(context, event):
    """Raycaster for dragging"""
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for reg in area.regions:
                if reg.type == 'WINDOW':
                    region = reg
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    if hasattr(space, 'region_3d'):
                        rv3d = space.region_3d
    
    user32 = windll.user32
    screensize = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
    
    X= region.x
    Y= region.y
    top = screensize[1]

    win_x = bpy.context.window_manager.windows[0].x
    win_y = bpy.context.window_manager.windows[0].y

    flipped = top - (event['y'] + Y + win_y)
    
    coord = (event['x'] - win_x - X, flipped)

    view3d_utils.region_2d_to_location_3d
    
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    ray_target = ray_origin + view_vector

    def visible_objects_and_duplis():
        """Loop over (object, matrix) pairs (mesh only)"""

        for obj in context.visible_objects:
            if obj.type == 'MESH':
                if obj.modeling_cloth:    
                    yield (obj, obj.matrix_world.copy())

    def obj_ray_cast(obj, matrix):
        """Wrapper for ray casting that moves the ray into object space"""

        # get the ray relative to the object
        matrix_inv = matrix.inverted()
        ray_origin_obj = matrix_inv * ray_origin
        ray_target_obj = matrix_inv * ray_target
        ray_direction_obj = ray_target_obj - ray_origin_obj
        
        # cast the ray
        success, location, normal, face_index = obj.ray_cast(ray_origin_obj, ray_direction_obj)

        if success:
            return location, normal, face_index, ray_target
        else:
            return None, None, None, ray_target

    # cast rays and find the closest object
    best_length_squared = -1.0
    best_obj = None

    for obj, matrix in visible_objects_and_duplis():
        hit, normal, face_index, target = obj_ray_cast(obj, matrix)
        extra_data['target'] = target
        if hit is not None:

            
            hit_world = matrix * hit
            length_squared = (hit_world - ray_origin).length_squared

            if best_obj is None or length_squared < best_length_squared:
                best_length_squared = length_squared
                best_obj = obj
                vidx = [v for v in obj.data.polygons[face_index].vertices]
                vert = obj.data.shape_keys.key_blocks['modeling cloth key'].data
            if best_obj is not None:    

                if extra_data['clicked']:    
                    #extra_data['move'] = np.array([0.0, 0.0, 0.0])
                    extra_data['matrix'] = matrix.inverted()
                    data[best_obj.name].clicked = True
                    extra_data['stored_mouse'] = np.copy(target)
                    extra_data['vidx'] = vidx
                    extra_data['stored_vidx'] = np.array([vert[v].co for v in extra_data['vidx']])
                    extra_data['clicked'] = False
                    
    if extra_data['stored_mouse'] is not None:
        move = np.array(extra_data['target'] * extra_data['matrix']) - extra_data['stored_mouse']
        extra_data['move'] = move

                   
# dragger===
class ModelingClothDrag(bpy.types.Operator):
    """Modal ray cast for dragging"""
    bl_idname = "view3d.modeling_cloth_drag"
    bl_label = "Modeling Cloth Drag"
    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self):
        bpy.ops.object.select_all(action='DESELECT')    
        extra_data['hit'] = None
        extra_data['clicked'] = False
        extra_data['stored_mouse'] = None
        extra_data['vidx'] = None
        extra_data['new_click'] = True
        extra_data['target'] = None
        for i in data:
            data[i].clicked = False
        
    def modal(self, context, event):
        bpy.context.window.cursor_set("HAND")
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            # allow navigation
            return {'PASS_THROUGH'}
        elif event.type == 'MOUSEMOVE':
            pos = queryMousePosition()            
            main_drag(context, pos)
            return {'RUNNING_MODAL'}
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # when I click, If I have a hit, store the hit on press
            extra_data['clicked'] = True
            extra_data['vidx'] = []
            
        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            extra_data['clicked'] = False
            extra_data['stored_mouse'] = None
            extra_data['vidx'] = None
            extra_data['new_click'] = True
            extra_data['target'] = None
            for i in data:
                data[i].clicked = False

            
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            extra_data['drag_alert'] = False
            extra_data['clicked'] = False
            extra_data['hit'] = None
            bpy.context.window.cursor_set("DEFAULT")
            extra_data['stored_mouse'] = None
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        cloth_objects = False
        cloths = [i for i in bpy.data.objects if i.modeling_cloth] # so we can select an empty and keep the settings menu up
        if len(cloths) > 0:
            cloth_objects = True        
            extra_data['drag_alert'] = True
            
        if context.space_data.type == 'VIEW_3D' and cloth_objects:
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            extra_data['drag_alert'] = False
            extra_data['stored_mouse'] = None
            bpy.context.window.cursor_set("DEFAULT")
            return {'CANCELLED'}


# drag===================================End
# drag===================================End



class DeletePins(bpy.types.Operator):
    """Delete modeling cloth pins and clear pin list for current object"""
    bl_idname = "object.delete_modeling_cloth_pins"
    bl_label = "Delete Modeling Cloth Pins"
    bl_options = {'REGISTER', 'UNDO'}    
    def execute(self, context):

        ob = get_last_object() # returns tuple with list and last cloth objects or None
        if ob is not None:
            print(data[ob[1].name].pin_list, 'at start')            
            l_copy = data[ob[1].name].pin_list[:]
            h_copy = data[ob[1].name].hook_list[:]
            for i in range(len(data[ob[1].name].hook_list)):
                if data[ob[1].name].hook_list[i].select:
                    bpy.data.objects.remove(data[ob[1].name].hook_list[i])
                    l_copy.remove(data[ob[1].name].pin_list[i]) 
                    h_copy.remove(data[ob[1].name].hook_list[i]) 
            
            data[ob[1].name].pin_list = l_copy
            data[ob[1].name].hook_list = h_copy
            print(data[ob[1].name].pin_list, 'after')        

        bpy.context.scene.objects.active = ob[1]
        return {'FINISHED'}


class SelectPins(bpy.types.Operator):
    """Select modeling cloth pins for current object"""
    bl_idname = "object.select_modeling_cloth_pins"
    bl_label = "Select Modeling Cloth Pins"
    bl_options = {'REGISTER', 'UNDO'}    
    def execute(self, context):
        ob = get_last_object() # returns list and last cloth objects or None
        if ob is not None:
            bpy.ops.object.select_all(action='DESELECT')
            for i in data[ob[1].name].hook_list:
                i.select = True

        return {'FINISHED'}


class PinSelected(bpy.types.Operator):
    """Add pins to verts selected in edit mode"""
    bl_idname = "object.modeling_cloth_pin_selected"
    bl_label = "Modeling Cloth Pin Selected"
    bl_options = {'REGISTER', 'UNDO'}    
    def execute(self, context):
        ob = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        sel = [i.index for i in ob.data.vertices if i.select]
                
        name = ob.name
        matrix = ob.matrix_world.copy()
        for v in sel:    
            e = bpy.data.objects.new('modeling_cloth_pin', None)
            bpy.context.scene.objects.link(e)
            if ob.active_shape_key is None:    
                closest = matrix * ob.data.vertices[v].co# * matrix
            else:
                closest = matrix * ob.active_shape_key.data[v].co# * matrix
            e.location = closest #* matrix
            e.show_x_ray = True
            e.select = True
            e.empty_draw_size = .1
            data[name].pin_list.append(v)
            data[name].hook_list.append(e)            
            ob.select = False
        bpy.ops.object.mode_set(mode='EDIT')       
        
        return {'FINISHED'}


class UpdataPinWeights(bpy.types.Operator):
    """Update Pin Weights"""
    bl_idname = "object.modeling_cloth_update_pin_group"
    bl_label = "Modeling Cloth Update Pin Weights"
    bl_options = {'REGISTER', 'UNDO'}        
    def execute(self, context):
        update_pin_group()
        return {'FINISHED'}


class GrowSource(bpy.types.Operator):
    """Grow Source Shape"""
    bl_idname = "object.modeling_cloth_grow"
    bl_label = "Modeling Cloth Grow"
    bl_options = {'REGISTER', 'UNDO'}        
    def execute(self, context):
        scale_source(1.02)
        return {'FINISHED'}


class ShrinkSource(bpy.types.Operator):
    """Shrink Source Shape"""
    bl_idname = "object.modeling_cloth_shrink"
    bl_label = "Modeling Cloth Shrink"
    bl_options = {'REGISTER', 'UNDO'}        
    def execute(self, context):
        scale_source(0.98)
        return {'FINISHED'}


class ResetShapes(bpy.types.Operator):
    """Reset Shapes"""
    bl_idname = "object.modeling_cloth_reset"
    bl_label = "Modeling Cloth Reset"
    bl_options = {'REGISTER', 'UNDO'}        
    def execute(self, context):
        reset_shapes()
        return {'FINISHED'}


class AddVirtualSprings(bpy.types.Operator):
    """Add Virtual Springs Between All Selected Vertices"""
    bl_idname = "object.modeling_cloth_add_virtual_spring"
    bl_label = "Modeling Cloth Add Virtual Spring"
    bl_options = {'REGISTER', 'UNDO'}        
    def execute(self, context):
        add_virtual_springs()
        return {'FINISHED'}


class RemoveVirtualSprings(bpy.types.Operator):
    """Remove Virtual Springs Between All Selected Vertices"""
    bl_idname = "object.modeling_cloth_remove_virtual_spring"
    bl_label = "Modeling Cloth Remove Virtual Spring"
    bl_options = {'REGISTER', 'UNDO'}        
    def execute(self, context):
        add_virtual_springs(remove=True)
        return {'FINISHED'}


def create_properties():            

    bpy.types.Object.modeling_cloth = bpy.props.BoolProperty(name="Modeling Cloth", 
        description="For toggling modeling cloth", 
        default=False, update=init_cloth)

    bpy.types.Object.modeling_cloth_floor = bpy.props.BoolProperty(name="Modeling Cloth Floor", 
        description="Stop at floor", 
        default=False)

    bpy.types.Object.modeling_cloth_pause = bpy.props.BoolProperty(name="Modeling Cloth Pause", 
        description="Stop without removing data", 
        default=True, update=pause_update)

    bpy.types.Object.modeling_cloth_noise = bpy.props.FloatProperty(name="Modeling Cloth Noise", 
        description="Set the noise strength", 
        default=0.001, precision=4, min=0, max=1, update=refresh_noise)

    bpy.types.Object.modeling_cloth_noise_decay = bpy.props.FloatProperty(name="Modeling Cloth Noise Decay", 
        description="Multiply the noise by this value each iteration", 
        default=0.99, precision=4, min=0, max=1)#, update=refresh_noise_decay)

    bpy.types.Object.modeling_cloth_spring_force = bpy.props.FloatProperty(name="Modeling Cloth Spring Force", 
        description="Set the spring force", 
        default=0.9, precision=4, min=0, max=1.3)#, update=refresh_noise)

    bpy.types.Object.modeling_cloth_gravity = bpy.props.FloatProperty(name="Modeling Cloth Gravity", 
        description="Modeling cloth gravity", 
        default=0.0, precision=4, min= -1, max=1)#, update=refresh_noise_decay)

    bpy.types.Object.modeling_cloth_iterations = bpy.props.IntProperty(name="Stiffness", 
        description="How stiff the cloth is", 
        default=1, min=1, max=50)#, update=refresh_noise_decay)

    bpy.types.Object.modeling_cloth_velocity = bpy.props.FloatProperty(name="Velocity", 
        description="Cloth keeps moving", 
        default=.9, min= -1.1, max=1.1, soft_min= -1, soft_max=1)#, update=refresh_noise_decay)

    bpy.types.Object.modeling_cloth_self_collision = bpy.props.BoolProperty(name="Modeling Cloth Self Collsion", 
        description="Toggle self collision", 
        default=False, update=collision_data_update)

    bpy.types.Object.modeling_cloth_self_collision_force = bpy.props.FloatProperty(name="recovery force", 
        description="Self colide faces repel", 
        default=.17, precision=4, min= -1.1, max=1.1, soft_min= 0, soft_max=1)

    bpy.types.Object.modeling_cloth_self_collision_margin = bpy.props.FloatProperty(name="Margin", 
        description="Self colide faces margin", 
        default=.08, precision=4, min= -1, max=1, soft_min= 0, soft_max=1)

    bpy.types.Object.modeling_cloth_self_collision_cy_size = bpy.props.FloatProperty(name="Cylinder size", 
        description="Self colide faces cylinder size", 
        default=1, precision=4, min= 0, max=4, soft_min= 0, soft_max=1.5)

    bpy.types.Object.modeling_cloth_inflate = bpy.props.FloatProperty(name="inflate", 
        description="add force to vertex normals", 
        default=0, precision=4, min= -10, max=10, soft_min= -1, soft_max=1)


    bpy.types.Object.modeling_cloth_sew = bpy.props.FloatProperty(name="sew", 
        description="add force to vertex normals", 
        default=0, precision=4, min= -10, max=10, soft_min= -1, soft_max=1)

    bpy.types.Scene.modeling_cloth_data_set = {} 
    bpy.types.Scene.modeling_cloth_data_set_extra = {} 

        
def remove_properties():            
    '''Drives to the grocery store and buys a sandwich'''
    del(bpy.types.Object.modeling_cloth)
    del(bpy.types.Object.modeling_cloth_floor)
    del(bpy.types.Object.modeling_cloth_pause)
    del(bpy.types.Object.modeling_cloth_noise)    
    del(bpy.types.Object.modeling_cloth_noise_decay)
    del(bpy.types.Object.modeling_cloth_spring_force)
    del(bpy.types.Object.modeling_cloth_gravity)        
    del(bpy.types.Object.modeling_cloth_iterations)
    del(bpy.types.Object.modeling_cloth_velocity)
    del(bpy.types.Object.modeling_cloth_inflate)
    del(bpy.types.Object.modeling_cloth_sew)

    # self collision
    del(bpy.types.Object.modeling_cloth_self_collision)    
    del(bpy.types.Object.modeling_cloth_self_collision_cy_size)    
    del(bpy.types.Object.modeling_cloth_self_collision_force)    
    del(bpy.types.Object.modeling_cloth_self_collision_margin)    

    # data storage
    del(bpy.types.Scene.modeling_cloth_data_set)
    del(bpy.types.Scene.modeling_cloth_data_set_extra)


class ModelingClothPanel(bpy.types.Panel):
    """Modeling Cloth Panel"""
    bl_label = "Modeling Cloth Panel"
    bl_idname = "Modeling Cloth"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Extended Tools"
    #gt_show = True
    
    def draw(self, context):
        status = False
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Modeling Cloth")
        ob = bpy.context.object
        cloths = [i for i in bpy.data.objects if i.modeling_cloth] # so we can select an empty and keep the settings menu up
        if len(cloths) > 0:                                        #
            status = extra_data['alert']
            if ob is not None:    
                if ob.type != 'MESH' or status:
                    ob = extra_data['last_object']                         #

        if ob is not None:
            if ob.type == 'MESH':
                col.prop(ob ,"modeling_cloth", text="Modeling Cloth", icon='SURFACE_DATA')               
                
                pause = 'PAUSE'
                if ob.modeling_cloth_pause:
                    pause = 'PLAY'
                
                if ob.modeling_cloth:
                    col = layout.column(align=True)
                    col.scale_y = 2.0
                    col.prop(ob ,"modeling_cloth_pause", text=pause, icon=pause)               
                    col.operator("object.modeling_cloth_reset", text="Reset")                    
                    col = layout.column(align=True)
                    col.scale_y = 2.0
                    col.alert = extra_data['drag_alert']
                    col.operator("view3d.modeling_cloth_drag", text="Grab")
                    col = layout.column(align=True)
                    
                col.prop(ob ,"modeling_cloth_iterations", text="Iterations")#, icon='OUTLINER_OB_LATTICE')               
                col.prop(ob ,"modeling_cloth_spring_force", text="Stiffness")#, icon='OUTLINER_OB_LATTICE')               
                col.prop(ob ,"modeling_cloth_noise", text="Noise")#, icon='PLAY')               
                col.prop(ob ,"modeling_cloth_noise_decay", text="Decay Noise")#, icon='PLAY')               
                col.prop(ob ,"modeling_cloth_gravity", text="Gravity")#, icon='PLAY')        
                col.prop(ob ,"modeling_cloth_inflate", text="Inflate")#, icon='PLAY')        
                col.prop(ob ,"modeling_cloth_sew", text="Sew Force")#, icon='PLAY')        
                col.prop(ob ,"modeling_cloth_velocity", text="Velocity")#, icon='PLAY')        
                col.prop(ob ,"modeling_cloth_floor", text="Floor")#, icon='PLAY')        
                col = layout.column(align=True)
                col.scale_y = 1.5
                col.alert = status
                if ob.modeling_cloth:    
                    if ob.mode == 'EDIT':
                        col.operator("object.modeling_cloth_pin_selected", text="Pin Selected")
                        col = layout.column(align=True)
                        col.operator("object.modeling_cloth_add_virtual_spring", text="Add Virtual Springs")
                        col.operator("object.modeling_cloth_remove_virtual_spring", text="Remove Selected")
                    else:
                        col.operator("view3d.modeling_cloth_pin", text="Create Pins")
                    col = layout.column(align=True)
                    col.operator("object.select_modeling_cloth_pins", text="Select Pins")
                    col.operator("object.delete_modeling_cloth_pins", text="Delete Pins")
                    col.operator("object.modeling_cloth_grow", text="Grow Source")
                    col.operator("object.modeling_cloth_shrink", text="Shrink Source")
                    col = layout.column(align=True)
                    col.prop(ob ,"modeling_cloth_self_collision", text="Self Collision")#, icon='PLAY')        
                    col.prop(ob ,"modeling_cloth_self_collision_force", text="Repel")#, icon='PLAY')        
                    col.prop(ob ,"modeling_cloth_self_collision_margin", text="Margin")#, icon='PLAY')        
                    col.prop(ob ,"modeling_cloth_self_collision_cy_size", text="Cylinder Size")#, icon='PLAY')        


# ============================================================================================
                    col = layout.column(align=True)
                    col.label('Collision Series')
                    col.operator("object.modeling_cloth_collision_series", text="Paperback")
                    col.operator("object.modeling_cloth_collision_series_kindle", text="Kindle")
                    col.operator("object.modeling_cloth_donate", text="Donate")


class CollisionSeries(bpy.types.Operator):
    """Support my addons by checking out my awesome sci fi books"""
    bl_idname = "object.modeling_cloth_collision_series"
    bl_label = "Modeling Cloth Collision Series"
        
    def execute(self, context):
        collision_series()
        return {'FINISHED'}


class CollisionSeriesKindle(bpy.types.Operator):
    """Support my addons by checking out my awesome sci fi books"""
    bl_idname = "object.modeling_cloth_collision_series_kindle"
    bl_label = "Modeling Cloth Collision Series Kindle"
        
    def execute(self, context):
        collision_series(False)
        return {'FINISHED'}


class Donate(bpy.types.Operator):
    """Support my addons by donating"""
    bl_idname = "object.modeling_cloth_donate"
    bl_label = "Modeling Cloth Donate"

        
    def execute(self, context):
        collision_series(False, False)
        self.report({'INFO'}, 'Paypal, The3dAdvantage@gmail.com')
        return {'FINISHED'}


def collision_series(paperback=True, kindle=True):
    import webbrowser
    import imp
    if paperback:    
        webbrowser.open("https://www.createspace.com/6043857")
        imp.reload(webbrowser)
        webbrowser.open("https://www.createspace.com/7164863")
        return
    if kindle:
        webbrowser.open("https://www.amazon.com/Resolve-Immortal-Flesh-Collision-Book-ebook/dp/B01CO3MBVQ")
        imp.reload(webbrowser)
        webbrowser.open("https://www.amazon.com/Formulacrum-Collision-Book-Rich-Colburn-ebook/dp/B0711P744G")
        return
    webbrowser.open("https://www.paypal.com/donate/?token=G1UymFn4CP8lSFn1r63jf_XOHAuSBfQJWFj9xjW9kWCScqkfYUCdTzP-ywiHIxHxYe7uJW&country.x=US&locale.x=US")

# ============================================================================================    
    


def register():
    create_properties()
    bpy.utils.register_class(ModelingClothPanel)
    bpy.utils.register_class(ModelingClothPin)
    bpy.utils.register_class(ModelingClothDrag)
    bpy.utils.register_class(DeletePins)
    bpy.utils.register_class(SelectPins)
    bpy.utils.register_class(PinSelected)
    bpy.utils.register_class(GrowSource)
    bpy.utils.register_class(ShrinkSource)
    bpy.utils.register_class(ResetShapes)
    bpy.utils.register_class(UpdataPinWeights)
    bpy.utils.register_class(AddVirtualSprings)
    bpy.utils.register_class(RemoveVirtualSprings)
    
    
    bpy.utils.register_class(CollisionSeries)
    bpy.utils.register_class(CollisionSeriesKindle)
    bpy.utils.register_class(Donate)


def unregister():
    remove_properties()
    bpy.utils.unregister_class(ModelingClothPanel)
    bpy.utils.unregister_class(ModelingClothPin)
    bpy.utils.unregister_class(ModelingClothDrag)
    bpy.utils.unregister_class(DeletePins)
    bpy.utils.unregister_class(SelectPins)
    bpy.utils.unregister_class(PinSelected)
    bpy.utils.unregister_class(GrowSource)
    bpy.utils.unregister_class(ShrinkSource)
    bpy.utils.unregister_class(ResetShapes)
    bpy.utils.unregister_class(UpdataPinWeights)
    bpy.utils.unregister_class(AddVirtualSprings)
    bpy.utils.unregister_class(RemoveVirtualSprings)
    
    
    bpy.utils.unregister_class(CollisionSeries)
    bpy.utils.unregister_class(CollisionSeriesKindle)
    bpy.utils.unregister_class(Donate)
    
    
if __name__ == "__main__":
    register()

    # testing!!!!!!!!!!!!!!!!
    #generate_collision_data(bpy.context.object)
    # testing!!!!!!!!!!!!!!!!


    
    for i in bpy.data.objects:
        i.modeling_cloth = False
    
    for i in bpy.app.handlers.scene_update_post:
        if i.__name__ == 'modeling_cloth_handler':
            bpy.app.handlers.scene_update_post.remove(i)
