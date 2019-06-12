# You are at the top. If you attempt to go any higher
#   you will go beyond the known limits of the code
#   universe where there are most certainly monsters

# to do:
#  self colisions
    # maybe do dynamic margins for when cloth is moving fast

#  object collisions

#  Other:
#  option to cache animation?
#  Custom Source shape option for animated shapes

# Sewing
# Could create super sewing that doesn't use edges but uses scalars along the edge to place virtual points
#   sort of a barycentric virtual spring. Could even use it to sew to faces if I can think of a ui for where on the face.
# On an all triangle mesh, where sew edges come together there are long strait lines. This probably causes those edges to fold.
#   in other words... creating diagonal springs between these edges will not solve the fold problem. Bend spring could do this.

#  Bend springs:
#  need to speed things up
#  When faces have various sizes, the forces don't add up
#  add curl by shortening bending springs on one axis or diagonal
#  independantly scale bending springs and structural to create buckling

# specific to using as blender addon:


"""Bug list"""
# if a subsurf modifier is on the cloth, the grab tool freaks

# updates to addons
# https://www.youtube.com/watch?v=Mjy-zGG3Wk4


bl_info = {
    "name": "Modeling Cloth",
    "author": "Rich Colburn, email: the3dadvantage@gmail.com",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Extended Tools > Modeling Cloth",
    "description": "Maintains the surface area of an object so it behaves like cloth",
    "warning": "Your future self is planning to travel back in time to kill you.",
    "wiki_url": "",
    "category": '3D View'}


import bpy
import bmesh
import numpy as np
from numpy import newaxis as nax
from bpy_extras import view3d_utils
import time
import mathutils

#enable_numexpr = True
enable_numexpr = False
if enable_numexpr:
    import numexpr as ne


you_have_a_sense_of_humor = False
#you_have_a_sense_of_humor = True
if you_have_a_sense_of_humor:
    import antigravity


def get_co(ob, arr=None, key=None): # key
    """Returns vertex coords as N x 3"""
    c = len(ob.data.vertices)
    if arr is None:    
        arr = np.zeros(c * 3, dtype=np.float32)
    if key is not None:
        ob.data.shape_keys.key_blocks[key].data.foreach_get('co', arr.ravel())        
        arr.shape = (c, 3)
        return arr
    ob.data.vertices.foreach_get('co', arr.ravel())
    arr.shape = (c, 3)
    return arr


def get_proxy_co(ob, arr, me):
    """Returns vertex coords with modifier effects as N x 3"""
    if arr is None:
        arr = np.zeros(len(me.vertices) * 3, dtype=np.float32)
        arr.shape = (arr.shape[0] //3, 3)    
    c = arr.shape[0]
    me.vertices.foreach_get('co', arr.ravel())
    arr.shape = (c, 3)
    return arr


def triangulate(me, cloth=None):
    """Requires a mesh. Returns an index array for viewing co as triangles"""
    obm = bmesh.new()
    obm.from_mesh(me)        
    bmesh.ops.triangulate(obm, faces=obm.faces)
    count = len(obm.faces)
    tri_idx = np.array([[v.index for v in f.verts] for f in obm.faces])
    
    # cloth can be the cloth object. Adds data to the class for the bend springs
    # Identify bend spring groups. Each edge gets paired with two points on tips of tris around edge    
    # Restricted to edges with two linked faces on a triangulated version of the mesh

    #"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    #cloth = None #!!!!!!!!!!!!!!!!!!!!!
    #"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"    
    #print("new=======================================")
    if cloth is not None:
        link_ed = [e for e in obm.edges if len(e.link_faces) == 2]
        # exclude pinned: (pin bool is those verts that are not pinned)
        
        # for each edge...
        # if either of the link faces of that edge
        # contain a vert that is not pinned
        # we keep that edge.


        # get the two verts from the link faces
        fv = np.array([[[v.index for v in f.verts] for f in e.link_faces] for e in link_ed])
        fv.shape = (fv.shape[0],6)        
        
        pindex = np.arange(cloth.pin_bool.shape[0])[cloth.pin_bool]
        sub_bend = [np.any(np.in1d(fv[i], pindex)) for i in range(fv.shape[0])]
        
    
        #print(sub_bend)
        link_ed = np.array(link_ed)[sub_bend]
        
        #print(sub_bend, "this is sub bend")
        
        #print(cloth.pin_bool, "did this work???")
        
        cloth.bend_eidx = np.array([[e.verts[0].index, e.verts[1].index] for e in link_ed])
        fv = np.array([[[v.index for v in f.verts] for f in e.link_faces] for e in link_ed])
        fv.shape = (fv.shape[0],6)
        cloth.bend_tips = np.array([[idx for idx in fvidx if idx not in e] for e, fvidx in zip(cloth.bend_eidx, fv)])

    
    obm.free()
    
    return tri_idx


def tri_normals_in_place(object, tri_co, start=False):    
    """Takes N x 3 x 3 set of 3d triangles and 
    generates unit normals and origins in the class"""
    object.origins = tri_co[:,0]
    object.cross_vecs = tri_co[:,1:] - object.origins[:, nax]
    object.normals = np.cross(object.cross_vecs[:,0], object.cross_vecs[:,1])
    object.nor_dots = np.einsum("ij, ij->i", object.normals, object.normals)
    if start:
        object.source_area = np.sqrt(object.nor_dots)
        object.source_total_area = np.sum(np.sqrt(object.nor_dots))
    object.normals /= np.sqrt(object.nor_dots)[:, nax]


def get_tri_normals(tr_co):
    """Takes N x 3 x 3 set of 3d triangles and 
    returns vectors around the triangle,
    non-unit normals and origins"""
    origins = tr_co[:,0]
    cross_vecs = tr_co[:,1:] - origins[:, nax]
    return cross_vecs, np.cross(cross_vecs[:,0], cross_vecs[:,1]), origins


def closest_points_edge(vec, origin, p):
    """Returns the location of the points on the edge,
    p is an Nx3 vector array"""
    vec2 = p - origin
    d = (vec2 @ vec) / (vec @ vec)
    cp = vec * d[:, nax]
    return cp, d


def proxy_in_place(object, me):
    """Overwrite vert coords with modifiers in world space"""
    me.vertices.foreach_get('co', object.co.ravel())
    object.co = apply_transforms(object.ob, object.co)


def apply_rotation(object):
    """When applying vectors such as normals we only need
    to rotate"""
    m = np.array(object.ob.matrix_world)
    mat = m[:3, :3].T
    object.v_normals = object.v_normals @ mat
    

def proxy_v_normals_in_place(object, world=True, me=None):
    """Overwrite vert coords with modifiers in world space"""
    me.vertices.foreach_get('normal', object.v_normals.ravel())
    if world:    
        apply_rotation(object)


def proxy_v_normals(ob, me):
    """Overwrite vert coords with modifiers in world space"""
    arr = np.zeros(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get('normal', arr)
    arr.shape = (arr.shape[0] //3, 3)
    m = np.array(ob.matrix_world, dtype=np.float32)    
    mat = m[:3, :3].T # rotates backwards without T
    return arr @ mat


def apply_transforms(ob, co):
    """Get vert coords in world space"""
    m = np.array(ob.matrix_world, dtype=np.float32)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


def apply_in_place(ob, arr, cloth):
    """Overwrite vert coords in world space"""
    m = np.array(ob.matrix_world, dtype=np.float32)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    arr[:] = arr @ mat + loc


def applied_key_co(ob, arr=None, key=None):
    """Get vert coords in world space"""
    c = len(ob.data.vertices)
    if arr is None:
        arr = np.zeros(c * 3, dtype=np.float32)
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', arr)
    arr.shape = (c, 3)
    m = np.array(ob.matrix_world)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


def revert_transforms(ob, co):
    """Set world coords on object. 
    Run before setting coords to deal with object transforms
    if using apply_transforms()"""
    m = np.linalg.inv(ob.matrix_world)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc  


def revert_in_place(ob, co):
    """Revert world coords to object coords in place."""
    m = np.linalg.inv(ob.matrix_world)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    co[:] = co @ mat + loc


def revert_rotation(ob, co):
    """When reverting vectors such as normals we only need
    to rotate"""
    m = np.array(ob.matrix_world)
    mat = m[:3, :3] # rotates backwards without T
    return co @ mat


def get_last_object():
    """Finds cloth objects for keeping settings active
    while selecting other objects like pins"""
    cloths = [i for i in bpy.data.objects if i.modeling_cloth] # so we can select an empty and keep the settings menu up
    if "object" not in dir(bpy.context):
        return
    
    if bpy.context.object is None:
        return
    
    if bpy.context.active_object.modeling_cloth:
        return cloths, bpy.context.active_object
    
    if len(cloths) > 0:
        ob = extra_data['last_object']
        return cloths, ob
    return None, None


def get_v_nor(ob, nor_arr):
    ob.data.vertices.foreach_get('normal', nor_arr.ravel())
    return nor_arr


def closest_point_edge(e1, e2, p):
    '''Returns the location of the point on the edge'''
    vec1 = e2 - e1
    vec2 = p - e1
    d = np.dot(vec2, vec1) / np.dot(vec1, vec1)
    cp = e1 + vec1 * d 
    return cp


def get_weights(ob, name):
    """Returns a numpy array containing the weights of
    the group as indicated by the string name. If there
    is no such group returns weights of zero."""
    v_count = len(ob.data.vertices)
    arr = np.zeros(v_count, dtype=np.float32)
    if name not in ob.vertex_groups:
        return arr    
    
    ind = ob.vertex_groups[name].index

    for i in range(v_count):
        if len(ob.data.vertices[i].groups) > 0:
            for j in ob.data.vertices[i].groups:
                if j.group == ind:
                    arr[i] = j.weight    
    return arr


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


def get_extend_springs(cloth, extend_springs=False):
    ob = cloth.ob
    obm = get_bmesh(ob)
    obm.edges.ensure_lookup_table()
    obm.verts.ensure_lookup_table()
    obm.faces.ensure_lookup_table()
    v_count = len(obm.verts)

    sew = np.array([len(i.link_faces)==0 for i in obm.edges])

    # get linear edges
    e_count = len(obm.edges)
    eidx = np.zeros(e_count * 2, dtype=np.int32)

    ob.data.edges.foreach_get('vertices', eidx)
    eidx.shape = (e_count, 2)
    sew_eidx = eidx[sew]        
    pure = eidx[~sew]
    
    # deal with sew verts connected to more than one edge
    s_t_rav = sew_eidx.T.ravel()
    s_uni, s_inv, s_counts = np.unique(s_t_rav, return_inverse=True, return_counts=True)
    s_multi = s_counts > 1
    
    multi_groups = None
    if np.any(s_counts):
        multi_groups = []
        ls = sew_eidx[:,0]
        rs = sew_eidx[:,1]
        
        for i in s_uni[s_multi]:
            gr = np.array([i])
            gr = np.append(gr, ls[rs==i])
            gr = np.append(gr, rs[ls==i])
            multi_groups.append(gr)

    uniidx = []

    for i in obm.verts:
        faces = i.link_faces
        f_verts = [[v for v in f.verts if v != i] for f in faces]
        lv = np.hstack(f_verts)
        for v in lv:
            uniidx.append([i.index, v.index])
    
    flip = np.sort(uniidx, axis=1)    
    uni = np.empty(shape=(0,2), dtype=np.int32)

    for i in range(v_count):
        this = flip[flip[:,0] == i]
        if this.shape[0] > 0:
            idx = this[np.unique(this[:,1], return_index=True)[1]]
            uni = np.append(uni, idx, axis=0)
    

    if extend_springs:
        extend = []
        pure = np.array(uniidx)
        for i in range(pure.shape[0]):  
            this = pure[pure[:,0] == i]
            if this.shape[0] > 0:    
                other = this[:,1]
                for j in other:

                    faces = obm.verts[j].link_faces
                    f_verts = [[v for v in f.verts if (v.index != j) & (v.index !=i)] for f in faces]
                    lv = np.hstack(f_verts)
                    for v in lv:
                        extend.append([i, v.index])

        extend = np.array(extend)
        e_flip = np.sort(extend, axis=1)    
        e_uni = np.empty(shape=(0,2), dtype=np.int32)

        for i in range(v_count):
            this = e_flip[e_flip[:,0] == i]
            if this.shape[0] > 0:
                idx = this[np.unique(this[:,1], return_index=True)[1]]
                e_uni = np.append(e_uni, idx, axis=0)
                
        cloth.eidx = e_uni
        return

    cloth.eidx = uni
    cloth.sew_edges = sew_eidx
    cloth.multi_sew = multi_groups    
    cloth.pure_eidx = pure #for experimental edge self collisions
        
    #return uni, e_uni, sew_eidx, multi_groups, pure
        

def get_unique_diagonal_edges(ob):
    """Creates a unique set of diagonal edges"""
    diag_eidx = []
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
                    skip = fv + 1 # this lets us skip the overlap so we don't have mirror duplicates
                roller = np.roll(f_verts, fv)
                for r in roller[skip:-1]:
                    diag_eidx.append([roller[0], r])

        start += fv_count
        
    return diag_eidx


def add_virtual_springs(remove=False):
    lo = get_last_object()
    if lo is None:
        return
    if lo[1] is None:
        return
    cloth = data[lo[1].name]
    obm = get_bmesh()
    obm.verts.ensure_lookup_table()
    count = len(obm.verts)
    idxer = np.arange(count, dtype=np.int32)
    sel = np.array([v.select for v in obm.verts])    
    selected = idxer[sel]

    if remove:
        ls = cloth.virtual_springs[:, 0]
        
        in_sel = np.in1d(ls, idxer[sel])

        deleter = np.arange(ls.shape[0], dtype=np.int32)[in_sel]
        reduce = np.delete(cloth.virtual_springs, deleter, axis=0)
        cloth.virtual_springs = reduce
        
        if cloth.virtual_springs.shape[0] == 0:
            cloth.virtual_springs.shape = (0, 2)
        return

    existing = np.append(cloth.eidx, cloth.virtual_springs, axis=0)
    flip = existing[:, ::-1]
    existing = np.append(existing, flip, axis=0)
    ls = existing[:,0]
        
    springs = []
    for i in idxer[sel]:

        # to avoid duplicates:
        # where this vert occurs on the left side of the existing spring list
        v_in = existing[i == ls]
        v_in_r = v_in[:,1]
        not_in = selected[~np.in1d(selected, v_in_r)]
        idx_set = not_in[not_in != i]
        for sv in idx_set:
            springs.append([i, sv])
    virtual_springs = np.array(springs, dtype=np.int32)
    
    if virtual_springs.shape[0] == 0:
        virtual_springs.shape = (0, 2)
    
    cloth.virtual_springs = np.append(cloth.virtual_springs, virtual_springs, axis=0)
    # gets appended to eidx in the cloth_init function after calling get connected polys in case geometry changes


def generate_guide_mesh():
    """Makes the arrow that appears when creating pins"""
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
        bpy.context.collection.objects.link(mesh_ob)
        #mesh_ob.show_x_ray = True
    return mesh_ob


def create_giude():
    """Spawns the guide"""
    if 'ModelingClothPinGuide' in bpy.data.objects:
        mesh_ob = bpy.data.objects['ModelingClothPinGuide']
        return mesh_ob
    mesh_ob = generate_guide_mesh()
    bpy.context.view_layer.objects.active = mesh_ob
    bpy.ops.object.material_slot_add()
    if 'ModelingClothPinGuide' in bpy.data.materials:
        mat = bpy.data.materials['ModelingClothPinGuide']
    else:    
        mat = bpy.data.materials.new(name='ModelingClothPinGuide')
    #mat.use_transparency = True
    #mat.alpha = 0.35            
    #mat.emit = 2     
    #mat.game_settings.alpha_blend = 'ALPHA_ANTIALIASING'
    #mat.diffuse_color = (1, 1, 0)
    #mesh_ob.material_slots[0].material = mat
    return mesh_ob


def delete_giude():
    """Deletes the arrow"""
    if 'ModelingClothPinGuide' in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects['ModelingClothPinGuide'])
    if 'ModelingClothPinGuide' in bpy.data.meshes:        
        guide_mesh = bpy.data.meshes['ModelingClothPinGuide']
        guide_mesh.user_clear()
        bpy.data.meshes.remove(guide_mesh)


def update_source(cloth):
    # measure bend source if using dynamic source:
    cloth.source_angles = bend_springs(cloth, cloth.sco, None)
            
    # linear spring measure
    cloth.ob.data.shape_keys.key_blocks['modeling cloth source key'].data.foreach_get('co', cloth.sco.ravel())
    svecs = cloth.sco[cloth.eidx[:, 1]] - cloth.sco[cloth.eidx[:, 0]]
    cloth.sdots = np.einsum('ij,ij->i', svecs, svecs)
    

def scale_source(multiplier, cloth=None):
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
            
            # recalculate in cloth: (so you don't have to have dynamic source on)
            update_source(data[ob.name])
            

def reset_shapes(ob=None):
    """Sets the modeling cloth key to match the source key.
    Will regenerate shape keys if they are missing"""
    if ob is None:    
        if bpy.context.active_object.modeling_cloth:
            ob = bpy.context.active_object
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
    keys['Basis'].data.foreach_get('co', co)
    #co = applied_key_co(ob, None, 'modeling cloth source key')
    #keys['modeling cloth source key'].data.foreach_set('co', co)
    keys['modeling cloth key'].data.foreach_set('co', co)
    
    # reset the data stored in the class
    data[ob.name].vel[:] = 0
    co.shape = (co.shape[0]//3, 3)
    data[ob.name].co = co
    data[ob.name].vel_start = np.copy(co)
    
    keys['modeling cloth key'].mute = True
    keys['modeling cloth key'].mute = False


def get_spring_mix(ob, eidx):
    rs = []
    ls = []
    minrl = []
    mixy = np.zeros(len(ob.data.vertices))
    for i in range(len(ob.data.vertices)):
        x = np.sum(eidx[:,0] == i)    
        y = np.sum(eidx[:,1] == i)    
        mixy[i] += x
        mixy[i] += y
    
    for i in eidx:    
        r = eidx[eidx == i[1]].shape[0]
        l = eidx[eidx == i[0]].shape[0]
        rs.append (min(r,l))
        ls.append (min(r,l))
    mix = 1 / np.array(rs + ls) ** 1.2
    mix = np.array(rs + ls)

    return mix, mixy[:, nax]
        

def update_pin_group():
    """Updates the cloth data after changing mesh or vertex weight pins"""
    ob = get_last_object()[1]
    if ob.name in data:
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


def generate_wind(wind_vec, cloth):
    """Maintains a wind array and adds it to the cloth vel"""    
    
    tri_nor = cloth.normals #/ np.array(cloth.ob.scale) # non-unit calculated by tri_normals_in_place() per each triangle
    w_vec = revert_rotation(cloth.ob, wind_vec) / np.array(cloth.ob.scale)

    turb = cloth.ob.modeling_cloth_turbulence    
    if turb != 0: 
        w_vec += (np.random.random(3).astype(np.float32) - 0.5) * turb * np.mean(w_vec) * 4

    # only blow on verts facing the wind
    perp = np.abs(tri_nor @ (w_vec * np.array(cloth.ob.scale)))
    cloth.wind += w_vec
    cloth.wind *= perp[:, nax][:, nax]
    
    # reshape for add.at
    shape = cloth.wind.shape
    cloth.wind.shape = (shape[0] * 3, 3)
    
    cloth.wind *= cloth.tri_mix
    np.add.at(cloth.vel, cloth.tridex.ravel(), cloth.wind)
    cloth.wind.shape = shape


def generate_inflate(cloth):
    """Blow it up baby!"""    
    shape = cloth.inflate.shape    
    force = cloth.ob.modeling_cloth_inflate
    norms = cloth.normals
    #area = np.sqrt(cloth.nor_dots)
    #area = ((cloth.source_area ** 2) + cloth.source_area) / 2
    area = cloth.source_area# ** 2 #) + cloth.source_area) / 2
    total_area = cloth.source_area
    current_area = np.sum(area)
    div = current_area / total_area
    
    #return
    #bippy = (norms * area[:, nax]) * force# * cloth.tri_mix) * force
    bippy = (norms) * force# * cloth.tri_mix) * force
    
    #print(bippy.shape)
    

    #return
    
    this = np.tile(bippy, 3)# * force * cloth.tri_mix #* div# * .02# * div
    this.shape = (shape[0] * 3, 3)
    this *= cloth.tri_mix
    
    np.add.at(cloth.vel, cloth.tridex.ravel(), this)
    return
    
    if False:
        tri_nor = cloth.normals #* cloth.ob.modeling_cloth_inflate # non-unit calculated by tri_normals_in_place() per each triangle
        #tri_nor /= np.einsum("ij, ij->i", tri_nor, tri_nor)[:, nax]
        # reshape for add.at
        shape = cloth.inflate.shape
        
        cloth.inflate += tri_nor[:, nax] * cloth.ob.modeling_cloth_inflate# * cloth.nor_dots[:,nax]
        
        cloth.inflate.shape = (shape[0] * 3, 3)
        

        cloth.inflate *= (cloth.tri_mix)# * cloth.nor_dots)
        root = np.sqrt(cloth.nor_dots)
        sum = np.sum(root)
        div = cloth.source_area / sum
        cloth.inflate *= np.repeat((cloth.nor_dots), 3)[:,nax] #* div
        
        this = np.tile(cloth.normals * cloth.source_area, 3) * div
        this.shape = (shape[0] * 3, 3)
        #print(cloth.tri_mix.shape)
        
        np.add.at(cloth.vel, cloth.tridex.ravel(), this)
        cloth.inflate.shape = shape
        #cloth.inflate *= 0


def get_quat(rad, axis):
    theta = (rad * 0.5)
    w = np.cos(theta)
    q_axis = axis * np.sin(theta)[:, nax]
    return w, q_axis


def q_rotate(co, w, axis):
    """Takes an N x 3 numpy array and returns that array rotated around
    the axis by the angle in radians w. (standard quaternion)"""    
    move1 = np.cross(axis, co)
    move2 = np.cross(axis, move1)
    move1 *= w[:, nax]
    return co + (move1 + move2) * 2


def bend_springs(cloth, co, measure=None):
    bend_eidx, tips = cloth.bend_eidx, cloth.bend_tips
    
    # if we have no springs...
    if tips.shape[0] < 1:
        return
    
    tips_co = co[tips]
    
    bls, brs = bend_eidx[:,0], bend_eidx[:, 1]
    b_oris = co[bls]
    
    be_vecs = co[brs] - b_oris
    te_vecs = tips_co - b_oris[:, nax]

    bcp_dots = np.einsum('ij,ikj->ik', be_vecs, te_vecs)
    be_dots = np.einsum('ij,ij->i', be_vecs, be_vecs)
    b_div = np.nan_to_num(bcp_dots / be_dots[:, nax])
    
    tcp = be_vecs[:, nax] * b_div[:, :, nax]
    
    # tip vecs from cp
    tcp_vecs = te_vecs - tcp
    tcp_dots = np.einsum('ijk,ijk->ij',tcp_vecs, tcp_vecs) 
    
    u_tcp_vecs = tcp_vecs / np.sqrt(tcp_dots)[:, :, nax]
    
    u_tcp_ls = u_tcp_vecs[:, 0]
    u_tcp_rs = u_tcp_vecs[:, 1]
    
    # dot of unit tri tips around axis
    angle_dot = np.einsum('ij,ij->i', u_tcp_ls, u_tcp_rs)
    
    #paralell = angle_dot < -.9999999
    
    angle = np.arccos(np.clip(angle_dot, -1, 1)) # values outside and arccos gives nan
    #angle = np.arccos(angle_dot) # values outside and arccos gives nan


    # get the angle sign
    tcp_cross = np.cross(u_tcp_vecs[:, 0], u_tcp_vecs[:, 1])
    sign = np.sign(np.einsum('ij,ij->i', be_vecs, tcp_cross))
    
    if measure is None:
        s = np.arccos(angle_dot)
        s *= sign
        s[angle_dot < -.9999999] = np.pi

        return s

    angle *= sign
    # rotate edges with quaternypoos
    u_be_vecs = be_vecs / np.sqrt(be_dots)[:, nax]
    b_dif = angle - measure
    
    l_ws, l_axes = get_quat(b_dif, u_be_vecs)
    r_ws, r_axes = l_ws, -l_axes
    
    # move tcp vecs so their origin is in the middle:
    #u_tcp_vecs *= 0.5    
    
    # should I rotate the unit vecs or the source?
    #   rotating the unit vecs here.
    
    stiff = cloth.ob.modeling_cloth_bend_stiff * 0.0057
    rot_ls = q_rotate(u_tcp_ls, l_ws, l_axes) 
    l_force = (rot_ls - u_tcp_ls) * stiff
    
    rot_rs = q_rotate(u_tcp_rs, r_ws, r_axes)    
    r_force = (rot_rs - u_tcp_rs) * stiff
    
    np.add.at(cloth.co, tips[:, 0], l_force)
    np.add.at(cloth.co, tips[:, 1], r_force)
    
    np.subtract.at(cloth.co, bend_eidx.ravel(), np.tile(r_force * .5, 2).reshape(r_force.shape[0] * 2, 3))
    np.subtract.at(cloth.co, bend_eidx.ravel(), np.tile(l_force * .5, 2).reshape(l_force.shape[0] * 2, 3))


# sewing functions ---------------->>>
def create_sew_edges():

    bpy.ops.mesh.bridge_edge_loops()
    bpy.ops.mesh.delete(type='ONLY_FACE')
    return
    # To do:
    #highlight a sew edge
    #compare vertex counts
    #subdivide to match counts
    #distribute and smooth back into mesh
    #create sew lines
# sewing functions ---------------->>>

    
class Cloth(object):
    pass


def create_instance(new=True):
    """Creates instance of cloth object with attributes needed for engine"""
    
    for i in bpy.data.meshes:
        if i.users == 0:
            bpy.data.meshes.remove(i)
    
    if new:
        cloth = Cloth()
        cloth.ob = bpy.context.active_object # based on what the user has as the active object
        cloth.pin_list = [] # these will not be moved by the engine
        cloth.hook_list = [] # these will be moved by hooks and updated to the engine
        cloth.virtual_springs = np.empty((0,2), dtype=np.int32) # so we can attach points to each other without regard to topology
        cloth.sew_springs = [] # edges with no faces attached can be set to shrink
    
    else: # if we set a modeling cloth object and have something else selected, the most recent object will still have it's settings exposed in the ui
        ob = get_last_object()[1]
        cloth = data[ob.name]
        cloth.ob = ob 
    
    bpy.context.view_layer.objects.active = cloth.ob
    cloth.idxer = np.arange(len(cloth.ob.data.vertices), dtype=np.int32)
    # data only accesible through object mode
    mode = cloth.ob.mode
    if mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # data is read from a source shape and written to the display shape so we can change the target springs by changing the source shape
    cloth.name = cloth.ob.name
    if cloth.ob.data.shape_keys == None:
        cloth.ob.shape_key_add(name='Basis')    
    if 'modeling cloth source key' not in cloth.ob.data.shape_keys.key_blocks:
        cloth.ob.shape_key_add(name='modeling cloth source key')        
    if 'modeling cloth key' not in cloth.ob.data.shape_keys.key_blocks:
        cloth.ob.shape_key_add(name='modeling cloth key')        
        cloth.ob.data.shape_keys.key_blocks['modeling cloth key'].value=1
    cloth.count = len(cloth.ob.data.vertices)
    
    # we can set a large group's pin state using the vertex group. No hooks are used here
    if 'modeling_cloth_pin' not in cloth.ob.vertex_groups:
        cloth.ob.vertex_groups.new(name='modeling_cloth_pin')
    cloth.pin_weights = get_weights(cloth.ob, 'modeling_cloth_pin')
    
    # pin bool sets all weights more than zero as pinned
    on_off = False
    if on_off:    
        cloth.pin_bool = ~cloth.pin_weights.astype(np.bool)
    
    # for use with weights with different values
    w_bool = cloth.pin_weights == 1
    cloth.pin_bool = ~(w_bool)

    # Spring Relationships
    
    # Extend Springs
    get_extend_springs(cloth) # uni, e_uni, sew_eidx, multi_groups, pure
    
    if cloth.ob.modeling_cloth_extend_springs:
        get_extend_springs(cloth, extend_springs=True)
    
    # Virtual Springs
    if cloth.virtual_springs.shape[0] > 0:
        cloth.eidx = np.append(cloth.eidx, cloth.virtual_springs, axis=0)
    
    cloth.eidx_tiler = cloth.eidx.T.ravel()    

    #mixology, cloth.spring_counts = get_spring_mix(cloth.ob, cloth.eidx)
    #cloth.spring_counts = get_spring_mix(cloth.ob, cloth.eidx)
    
    pindexer = np.arange(cloth.count, dtype=np.int32)[cloth.pin_bool]
    unpinned = np.in1d(cloth.eidx_tiler, pindexer)
    #cloth.eidx_tiler = cloth.eidx_tiler[unpinned]    
    cloth.unpinned = unpinned
    
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
    
    
    cloth.self_col_vel = np.copy(co)
    cloth.weighted_average = np.copy(co)
    cloth.weighted_average[:] = 0
    cloth.weight_sum = np.zeros(co.shape[0], dtype=np.float32)
    
    cloth.v_normals = np.zeros(co.shape, dtype=np.float32)
    
    # blended pinning:
    cloth.weighted = (cloth.pin_weights < 1) & (cloth.pin_weights > 0)
    cloth.blended_co = cloth.co[cloth.weighted]
    cloth.blend_weights = cloth.pin_weights[cloth.weighted][:,nax]
    
    #noise---
    noise_zeros = np.zeros(cloth.count, dtype=np.float32)
    random = np.random.random(cloth.count).astype(np.float32)
    noise_zeros[:] = random
    cloth.noise = ((noise_zeros + -0.5) * cloth.ob.modeling_cloth_noise * 0.1)[:, nax]
    
    cloth.waiting = False
    cloth.clicked = False # for the grab tool
    
    # this helps with extra springs behaving as if they had more mass---->>>
    #cloth.mix = mixology[unpinned][:, nax]
    # -------------->>>

    # new self collisions:
    cloth.tridex = triangulate(cloth.ob.data, cloth)
    cloth.eidxer = np.arange(cloth.pure_eidx.shape[0])
    cloth.tridexer = np.arange(cloth.tridex.shape[0], dtype=np.int32)
    cloth.tri_co = cloth.co[cloth.tridex]
    tri_normals_in_place(cloth, cloth.tri_co, start=True) # non-unit normals
    # -------------->>>
    
    tri_uni, tri_inv, tri_counts = np.unique(cloth.tridex, return_inverse=True, return_counts=True)
    cloth.tri_mix = (1 / tri_counts[tri_inv])[:, nax]
    
    cloth.wind = np.zeros(cloth.tri_co.shape, dtype=np.float32)
    cloth.inflate = np.zeros(cloth.tri_co.shape, dtype=np.float32)

    bpy.ops.object.mode_set(mode=mode)
    
    # for use with a static source shape:
    cloth.source_angles = bend_springs(cloth, cloth.sco, None)
    svecs = cloth.sco[cloth.eidx[:, 1]] - cloth.sco[cloth.eidx[:, 0]]
    cloth.sdots = np.einsum('ij,ij->i', svecs, svecs)
    
    # softbody via mean
    cloth.soft_list = [i for i in bpy.data.objects if i.modeling_cloth_softbody_goal]
    cloth.soft_data = {}
    if cloth.soft_list:
        cloth.soft_move = np.zeros(cloth.co.shape[0] * 3, dtype=np.float32)
        cloth.soft_move.shape = (cloth.co.shape[0], 3)
        for i in cloth.soft_list:    
            soft_target = revert_transforms(cloth.ob, np.array(i.location))
            
            #cloth.soft_target = soft_target
            target_vecs = soft_target - co
            soft_dots = np.einsum('ij,ij->i', target_vecs, target_vecs)
            cloth.soft_data[i.name + 'soft_dots'] = soft_dots

            if 'soft_goal_' + i.name not in cloth.ob.vertex_groups:
                cloth.ob.vertex_groups.new(name='soft_goal_' + i.name)
            cloth.soft_data[i.name + 'v_weights'] = get_weights(cloth.ob, 'soft_goal_' + i.name)[:, nax]
            #print(co.shape, "shape of co")
            #print(cloth.soft_data[i.name + 'v_weights'].shape, "shape of weights")
    # for doing static cling
    #   cloth.col_idx = np.array([], dtype=np.int32)
    #   cloth.re_col = np.empty((0,3), dtype=np.float32)
    
    return cloth


def run_handler(cloth):
    # can run the simulation constantly or with frame changes during blender animation:
    #if cloth.ob.modeling_cloth_handler_frame | bpy.app.timers.is_registered():
    if True:
        # pause the cloth engine if the current cloth object is in edit mode
        if cloth.ob.mode == 'EDIT':
            cloth.waiting = True
        if cloth.waiting:    
            if cloth.ob.mode == 'OBJECT':
                update_pin_group()

        if not cloth.waiting:
            
            # add noise to the unpinned vertices
            cloth.co[cloth.pindexer] += cloth.noise[cloth.pindexer]
            cloth.noise *= cloth.ob.modeling_cloth_noise_decay

            # mix in vel before collisions and sewing
            cloth.co[cloth.pindexer] += cloth.vel[cloth.pindexer]
            cloth.vel_start[:] = cloth.co

            # measure source -------------------------->>>
            dynamic = cloth.ob.modeling_cloth_dynamic_source # can store for speedup if source shape is static
            # bend spring calculations:
            source_angles = cloth.source_angles
            if cloth.ob.modeling_cloth_bend_stiff != 0:
                # measure bend source if using dynamic source:
                if dynamic:    
                    source_angles = bend_springs(cloth, cloth.sco, None)
            
            # linear spring measure
            sdots = cloth.sdots
            if dynamic:    
                cloth.ob.data.shape_keys.key_blocks['modeling cloth source key'].data.foreach_get('co', cloth.sco.ravel())
                svecs = cloth.sco[cloth.eidx[:, 1]] - cloth.sco[cloth.eidx[:, 0]]
                sdots = np.einsum('ij,ij->i', svecs, svecs)            
            # ----------------------------------------->>>
            
            force = cloth.ob.modeling_cloth_spring_force * 0.5
            
            ers = cloth.eidx[:, 1]
            els = cloth.eidx[:, 0]

            for x in range(cloth.ob.modeling_cloth_iterations):    
                Ti = time.time()
                # bend spring calculations:

                if cloth.ob.modeling_cloth_bend_stiff != 0:
                    bend_springs(cloth, cloth.co, source_angles)

                # add pull
                vecs = cloth.co[ers] - cloth.co[els]
                dots = np.einsum('ij,ij->i', vecs, vecs)
                div = sdots / dots
                dist = np.sqrt(div)
                swap = vecs * dist[:, nax]
                    
                # weighted average of move
                move = vecs - swap
                w = np.sqrt(np.einsum('ij,ij->i', move, move))
                weights = np.tile(w, 2)
                
                loc_a = cloth.co[els] + swap
                loc_b = cloth.co[ers] - swap
                both = np.append(loc_b, loc_a, axis=0) * weights[:, nax]
                
                T = cloth.eidx_tiler
                                
                cloth.weight_sum[:] = 0                
                np.add.at(cloth.weight_sum, T, weights)

                cloth.weighted_average[:] = 0                
                np.add.at(cloth.weighted_average, T, both)

                final_loc = cloth.weighted_average / cloth.weight_sum[:, nax]
                bananers = np.isnan(final_loc)
                final_loc[bananers] = cloth.co[bananers]
                move = final_loc - cloth.co

                cloth.co[cloth.pin_bool] += move[cloth.pin_bool] * force

                # soft_goal------>>>
                if cloth.soft_list:
                    cloth.soft_move[:] = 0
                    for i in cloth.soft_list:
                        s_force = i.modeling_cloth_softgoal_strength
                        target = revert_transforms(cloth.ob, np.array(i.location))
                        soft_source_dots = cloth.soft_data[i.name + 'soft_dots']

                        target_vecs = target - cloth.co
                        dots = np.einsum('ij,ij->i', target_vecs, target_vecs)
                        div = np.nan_to_num(soft_source_dots / dots)
                        swap = target_vecs * np.sqrt(div)[:, nax]
                        
                        cloth.soft_move += (target_vecs - swap)# * cloth.soft_data[i.name + 'v_weights']# * s_force

                        if not i.modeling_cloth_softbody_fixed_goal:
                            spring_back = .7
                            i.location = np.array(i.location) - np.mean(cloth.soft_move, axis=0) * spring_back
                    
                    #print()
                    
                    cloth.co += cloth.soft_move * cloth.soft_data[i.name + 'v_weights'] * s_force                   
                    """
                    add soft mean target with vertex groups
                    """
                
                # for doing static cling
                #   cloth.co[cloth.col_idx] = cloth.re_col
                
                # move blended pinned towards source
                fallof = cloth.ob.modeling_cloth_vertex_pin_fallof
                blend_vecs = cloth.blended_co - cloth.co[cloth.weighted]
                v_move = blend_vecs * cloth.blend_weights * fallof
                
                cloth.co[cloth.weighted] = cloth.co[cloth.weighted] + v_move
                
                # move pinned back
                cloth.co[~cloth.pin_bool] = cloth.vel_start[~cloth.pin_bool]

                if len(cloth.pin_list) > 0:
                    hook_co = np.array([cloth.ob.matrix_world.inverted() @ i.matrix_world.to_translation() for i in cloth.hook_list])
                    cloth.co[cloth.pin_list] = hook_co
                
                # grab inside spring iterations
                if cloth.clicked: # for the grab tool
                    cloth.co[extra_data['vidx']] = np.array(extra_data['stored_vidx']) + np.array(+ extra_data['move'])   


            # refresh normals for inflate wind and self collisions
            cloth.tri_co = cloth.co[cloth.tridex]
            tri_normals_in_place(cloth, cloth.tri_co) # unit normals
            
            # add effects of velocity and Gravity to the vel array for later
            spring_dif = cloth.co - cloth.vel_start
            
            # gravity
            grav = (cloth.ob.modeling_cloth_gravity * .01) / cloth.ob.scale.z
            if grav != 0:
                xx = revert_rotation(cloth.ob, np.array([0, 0, grav])) #/ cloth.ob.scale.z
                cloth.vel += xx / cloth.ob.scale.z

            # inextensible calc:
            cloth.vel += spring_dif * 2

            # The amount of drag increases with speed. 
            # have to convert to to a range between 0 and 1
            #squared_move_dist = np.sqrt(np.einsum("ij, ij->i", cloth.vel, cloth.vel))
            squared_move_dist = np.einsum("ij, ij->i", cloth.vel, cloth.vel)
            squared_move_dist += 1
            cloth.vel *= (1 / (squared_move_dist / cloth.ob.modeling_cloth_velocity))[:, nax]
            #cloth.vel *= cloth.ob.modeling_cloth_velocity

            # wind:
            x = cloth.ob.modeling_cloth_wind_x
            y = cloth.ob.modeling_cloth_wind_y
            z = cloth.ob.modeling_cloth_wind_z
            wind_vec = np.array([x,y,z])
            check_wind = wind_vec != 0
            if np.any(check_wind):
                generate_wind(wind_vec / np.array(cloth.ob.scale), cloth)            

            # inflate
            inflate = cloth.ob.modeling_cloth_inflate
            if inflate != 0:
                generate_inflate(cloth)

            if cloth.ob.modeling_cloth_sew != 0:
                if len(cloth.sew_edges) > 0:
                    sew_edges = cloth.sew_edges
                    rs = cloth.co[sew_edges[:,1]]
                    ls = cloth.co[sew_edges[:,0]]
                    sew_vecs = (rs - ls) * 0.5 * cloth.ob.modeling_cloth_sew
                    cloth.co[sew_edges[:,1]] -= sew_vecs
                    cloth.co[sew_edges[:,0]] += sew_vecs
                    
                    # for sew verts with more than one sew edge
                    if cloth.multi_sew is not None:
                        for sg in cloth.multi_sew:
                            cosg = cloth.co[sg]
                            meanie = np.mean(cosg, axis=0)
                            sg_vecs = meanie - cosg
                            cloth.co[sg] += sg_vecs * cloth.ob.modeling_cloth_sew
                
            # floor --- (simple floor collision)
            if cloth.ob.modeling_cloth_floor:    
                floored = cloth.co[:,2] < 0        
                cloth.vel[:,2][floored] *= -1
                cloth.vel[floored] *= .1
                cloth.co[:, 2][floored] = 0
            # floor ---            
            
            #%%%%%%%%%%% for new self collisions!!!
            #testing = False
            #testing = True
            #if testing:
                #edge_self(cloth)
            #%%%%%%%%%%% for new self collisions!!!            
            
            # objects ---
            if cloth.ob.modeling_cloth_object_detect:
                if cloth.ob.modeling_cloth_self_collision:
                    self_collide(cloth)
                
                if extra_data['colliders'] is not None:
                    for i, val in extra_data['colliders'].items():
                        if val.ob != cloth.ob:
                            object_collide(cloth, val)

            cloth.co[~cloth.pin_bool] = cloth.vel_start[~cloth.pin_bool]
            
            if len(cloth.pin_list) > 0:
                cloth.co[cloth.pin_list] = hook_co
                cloth.vel[cloth.pin_list] = 0

            if cloth.clicked: # for the grab tool
                cloth.co[extra_data['vidx']] = np.array(extra_data['stored_vidx']) + np.array(+ extra_data['move'])


            cloth.ob.data.shape_keys.key_blocks['modeling cloth key'].data.foreach_set('co', cloth.co.ravel())

            cloth.ob.data.shape_keys.key_blocks['modeling cloth key'].mute = True
            cloth.ob.data.shape_keys.key_blocks['modeling cloth key'].mute = False


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ rethinking collisions:
def bold_new_collisions():
    """
    Point clouds with barycentric?
    Could use weighted average since points don't go
    outside faces.
    
    Could do broad phase first then only create point clouds
    on faces that need checks...
    
    If points are generated in advance they will deform when
    stretched creating gaps.
    On the other hand, if points 
    
    !!!
    Use spheres to detect collisions but use face normals
    to correct.
    Could use points on one side and faces on the other...
    So points on the cloth mesh distributed over the face
    checked against normals of collide mesh...
    !!!
    
    """    



    # decide:
    # 1. Generate points in tris or track
    
    #--------------------------
    """
    What I'm doing now isn't spheres at points
    Need to create a solver that acts like
    spheres over points.
    Need to treet points as spheres in 
    broad phase also
    
    Start with extra sphere at mean of each
    tri. It will behave different than
    othe points because whole face will move
    with mean collide. 
    
    Margin should be sphere
    size.
    
    The barycentric thing for inside tris
    might be replaced by weighted average
    to check inside tris.
    
    Either way, need to figure out spheres
    inside tris
    
    For extra accuraccy I could do something as
    simple as a tri inside a tri, so the
    mid of each edge and the mean
    
    """
    #--------------------------
    
    pass
    
    
    def weighted_av_barycentric_spheres():
        pass    



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



# %%%%%%%%%%%%%%%%%%%
def edges_edges(co, eidx_pairs, cloth):
    """Find the two points that describe the shortest vector
    between two edge segments that is orthagonal to both"""

    #bpy.data.objects['e'].location = co[1][1][0]
    #print("ran edges_edges...")
    
    #e0 = eidx_pairs[:, 0, 0]
    #e1 = eidx_pairs[:, 0, 1]
    #e2 = eidx_pairs[:, 1, 0]
    #e3 = eidx_pairs[:, 1, 1]
    #print(e0[:4])
    #print(e1[:4])
    #print(e2[:4])
    #print(e3[:4])

    e0 = co[:, 0, 0]
    e1 = co[:, 0, 1]
    e2 = co[:, 1, 0]
    e3 = co[:, 1, 1]    
    
    v0 = e1 - e0
    v1 = e3 - e2
    or_vec = e0 - e2

    cross0 = np.cross(v1, v0)

    v1d = np.einsum('ij,ij->i',v1, v1)
    d = np.einsum('ij,ij->i',v1, v0) / v1d
    cp = v1 * d[:, nax]
    normal = v0 - cp

    e_dot = np.einsum('ij,ij->i', normal, v0)
    e_n_dot = np.einsum('ij,ij->i', normal, or_vec)
    scale = e_n_dot / e_dot
    
    test1 = (scale < 0) & (scale > -1)
    
    p = (or_vec - v0 * scale[:, nax]) + e2
    
    cross_dot = np.einsum('ij,ij->i', cross0, cross0)
    
    d2 = np.einsum('ij,ij->i', or_vec, cross0) / cross_dot
    spit = cross0 * d2[:, nax]    
    
    #check2:
    this = p - e2

    test_dot = np.einsum('ij,ij->i', v1, this) / v1d
    test2 = (test_dot < 1) & (test_dot > 0)

    both = test1 & test2
    if np.any(both):
        # distance between edges:
        spits = spit[both]
        dif = np.sqrt(np.einsum('ij,ij->i',spits, spits))
        margin = 0.1
        spit_dist = dif < margin
        if np.any(spit_dist):    
            dif = dif[spit_dist]
            
            ed = eidx_pairs[both][spit_dist] 
            final_spit = spits[spit_dist]
            
            cut = (margin - dif) / 2
            
            u_spit = final_spit / dif[:, nax]
            
            move = u_spit * cut[:, nax]
            
            
            up = ed[:, 0]
            down = ed[:, 1]
            
            #cloth.co[up]
            move = move[:, nax]
            
            #print(cloth_co[up].shape, "shape of co")
            #print(cloth_co[down].shape, "shape of co")
            #print(up.shape, "up shape")
            #print(move.shape, "move shape")
            cloth.co[up] += move
            cloth.co[down] -= move
            cloth.vel[up] *= 0
            cloth.vel[down] *= 0

            
        
        #cloth_co
        #print(co.shape, "shape of co")    
        #print(eidx_pairs[both], "here I am")    
            #co[eidx_pairs[both]] += .01
        #print(dif, "how far apart")

        return p, p - spit
        

def edge_self(cloth):

    eidx = cloth.pure_eidx
    margin = cloth.ob.modeling_cloth_self_collision_margin

    edge_co = cloth.co[eidx]
    # %%%%%%%%
    
    tri_min = np.min(edge_co, axis=1) - margin
    tri_max = np.max(edge_co, axis=1) + margin   

    # begin every vertex co against every tri
    v_tris = e_per_e(cloth.co, tri_min, tri_max, cloth.idxer, cloth.eidxer, None, None, cloth)
    #print(eidx[v_tris][:4])
    #print(eidx[v_tris][:4].shape)
    #print(edge_co[v_tris][:4].shape)

    edges_edges(edge_co[v_tris], eidx[v_tris], cloth)
    
        

    return
    if v_tris is not None:
        tidx = v_tris
        
        
        """
        Each edge needs to 
        """
        
        """you are here"""
        """you are here"""
        """you are here"""
        """you are here"""
        
        u_norms = cloth.normals

        # don't check faces the verts are part of        
        check_neighbors = cidx[:, nax] == cloth.tridex[tidx]
        cull = np.any(check_neighbors, axis=1)
        cidx, tidx = cidx[~cull], tidx[~cull]
        # %%%%%%%%%%%%%%%%%%%%%
        ori = cloth.origins[tidx]
        nor = u_norms[tidx]
        vec2 = cloth.co[cidx] - ori
        
        d = np.einsum('ij, ij->i', nor, vec2) # nor is unit norms
        in_margin = (d > -margin) & (d < margin)
        # <<<--- Inside triangle check --->>>
        # will overwrite in_margin:
        cross_2 = cloth.cross_vecs[tidx][in_margin]
        inside_triangles(cross_2, vec2[in_margin], cloth.co, edge_co, cidx, tidx, nor, ori, in_margin, offset=0.0)
        
        if np.any(in_margin):
            # collision response --------------------------->>>
            t_in = tidx[in_margin]
            #tri_vel1 = np.mean(edge_co[t_in], axis=1)
            #tvel = np.mean(tri_vo[t_in], axis=1)
            #tvel = tri_vel1 - tri_vel2
            t_vel = np.mean(cloth.vel[cloth.tridex][t_in], axis=1)
            
            col_idx = cidx[in_margin] 
            d_in = d[in_margin]
    
            sign_margin = margin * np.sign(d_in) # which side of the face
            c_move = ((nor[in_margin] * d_in[:, nax]) - (nor[in_margin] * sign_margin[:, nax]))#) * -np.sign(d[in_margin])[:, nax]
            #c_move *= 1 / cloth.ob.modeling_cloth_grid_size
            #cloth.co[col_idx] -= ((nor[in_margin] * d_in[:, nax]) - (nor[in_margin] * sign_margin[:, nax]))#) * -np.sign(d[in_margin])[:, nax]
            cloth.co[col_idx] -= c_move #* .7
            #cloth.vel[col_idx] = 0
            cloth.vel[col_idx] = t_vel


def e_per_e(co, tri_min, tri_max, idxer, tridexer, c_peat=None, t_peat=None, cloth=None):
    """Finds edges"""

    #co_x, co_y, co_z = co[:, 0], co[:, 1], co[:, 2]
    
    subs = 7
    #subs = bpy.data.objects['Plane.002'].modeling_cloth_grid_size
    
    t_peat = e_z_grid(subs, tri_min[:, 2], tri_max[:, 2], tri_min[:, 0], tri_max[:, 0], tri_min[:, 1], tri_max[:, 1], cloth)
    #print(t_peat)
    return t_peat
    #print("new-------------")
    #print(c_peat)
    #print(t_peat)
    
    #if c_peat is None:
        #return
    # X
    # Step 1 check x_min (because we're N squared here we break it into steps)
    check_x_min = co_x[c_peat] > tri_min[:, 0][t_peat]
    c_peat = c_peat[check_x_min]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_x_min]

    # Step 2 check x max
    check_x_max = co_x[c_peat] < tri_max[:, 0][t_peat]
    c_peat = c_peat[check_x_max]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_x_max]
    
    # Y
    # Step 3 check y min    
    check_y_min = co_y[c_peat] > tri_min[:, 1][t_peat]
    c_peat = c_peat[check_y_min]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_y_min]

    # Step 4 check y max
    check_y_max = co_y[c_peat] < tri_max[:, 1][t_peat]
    c_peat = c_peat[check_y_max]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_y_max]

    # Z
    # Step 5 check z min    
    check_z_min = co_z[c_peat] > tri_min[:, 2][t_peat]
    c_peat = c_peat[check_z_min]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_z_min]

    # Step 6 check y max
    check_z_max = co_z[c_peat] < tri_max[:, 2][t_peat]
    c_peat = c_peat[check_z_max]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_z_max]    

    return idxer[c_peat], t_peat
    #return c_peat, t_peat

#%%%%%%%%%%%%%%%%%
def e_zxy_grid(tymin, tymax, subs, t, t_peat, c_peat, cloth):
    # create linespace grid between bottom and top of tri z
    #subs = 7
    t_min = np.min(tymin)
    t_max = np.max(tymax)
    divs = np.linspace(t_min, t_max, num=subs, dtype=np.float32)            
    
    # figure out which triangles and which co are in each section
    #co_bools = (co_y > divs[:-1][:, nax]) & (co_y < divs[1:][:, nax])
    tri_bools = (tymin < divs[1:][:, nax]) & (tymax > divs[:-1][:, nax])

    for i in tri_bools:
        if np.sum(i) > 0:
            #c3 = c[i]
            t3 = t[i]
            #print(t3, 'this is t3')
            c_tile = np.repeat(t3, t3.shape[0])
            t_tile = np.tile(t3, t3.shape[0])
            
            duper_bool = c_tile - t_tile != 0
            
            c_tile = c_tile[duper_bool]
            t_tile = t_tile[duper_bool]
            
            eidx = cloth.pure_eidx
            
            b1 = eidx[c_tile] == eidx[t_tile]
            b2 = eidx[c_tile][:, ::-1] == eidx[t_tile]
            
            #if len(c_peat) < 1:
                #print(c_tile, 'c_tile??')
                #print(t_tile, 't_tile??')
            #print("------------------")
            #print(eidx[c_tile])
            #print(eidx[t_tile])
            #print(b1 )
            #print(b2)
            #print('-----------------------------')
            final = np.any(b1 + b2, axis=1)
            #print("group##########")
            #print(c_tile[~final])
            #print(t_tile[~final])
            #print("group##########")            
            c_peat.append(c_tile[~final])
            t_peat.append(t_tile[~final])
            
            #du
            
#%%%%%%%%%%%%%%%%%%%%
def e_zx_grid(txmin, txmax, subs, t, t_peat, tymin, tymax, c_peat, cloth):
    # create linespace grid between bottom and top of tri z
    #subs = 7
    t_min = np.min(txmin)
    t_max = np.max(txmax)
    divs = np.linspace(t_min, t_max, num=subs, dtype=np.float32)            
    
    # figure out which triangles and which co are in each section
    #co_bools = (co_x > divs[:-1][:, nax]) & (co_x < divs[1:][:, nax])
    tri_bools = (txmin < divs[1:][:, nax]) & (txmax > divs[:-1][:, nax])

    for i in tri_bools:
        if np.sum(i) > 0:
            #c2 = c[i]
            t2 = t[i]
            
            e_zxy_grid(tymin[i], tymax[i], subs, t2, t_peat, c_peat, cloth)

#%%%%%%%%%%%%%%%%%%
def e_z_grid(subs, tzmin, tzmax, txmin, txmax, tymin, tymax, cloth):
    # create linespace grid between bottom and top of tri z
    #subs = 7
    t_min = np.min(tzmin)
    t_max = np.max(tzmax)
    divs = np.linspace(t_min, t_max, num=subs, dtype=np.float32)
            
    # figure out which triangles and which co are in each section
    tri_bools = (tzmin < divs[1:][:, nax]) & (tzmax > divs[:-1][:, nax])

    t_ranger = np.arange(tri_bools.shape[1])

    c_peat = []
    t_peat = []

    for i in tri_bools:
        if np.sum(i) > 0:
            t = t_ranger[i]
            e_zx_grid(txmin[i], txmax[i], subs, t, t_peat, tymin[i], tymax[i], c_peat, cloth)
            
    chs = np.hstack(c_peat)
    ths = np.hstack(t_peat)

    return eliminate_duplicate_pairs(chs, ths)


def eliminate_duplicate_pairs(ar1, ar2):
    """Eliminates duplicates
    and mirror duplicates.
    for example, [1,4], [4,1]
    or duplicate occurrences of [1,4]
    Returns an Nx2 array."""
    # join arrays:
    shape = ar1.shape[0]
    app = np.append(ar1, ar2)
    app.shape = (2, shape)
    # transpose and sort mirrors smaller on the left
    a = np.sort(app.T, axis=1)
    x = np.random.rand(a.shape[1])
    y = a.dot(x)
    unique, index = np.unique(y, return_index=True)
    return a[index]   


def edge_edge(e0, e1, e2, e3):
    """Find the two points that describe the shortest vector
    between two edge segments that is orthagonal to both"""
    v0 = e1 - e0
    v1 = e3 - e2
    or_vec = e0 - e2

    cross0 = np.cross(v1, v0)

    d = (v1 @ v0) / (v1 @ v1)
    cp = v1 * d
    normal = v0 - cp
    
    e_dot = normal @ v0
    e_n_dot = normal @ or_vec
    scale = e_n_dot / e_dot
    
    p = (or_vec - v0 * scale) + e2

    d = (or_vec @ cross0) / (cross0 @ cross0)
    spit = cross0 * d

    return p, p - spit

    
# %%%%%%%%%%%%%%%%%%%
# end experimental edge collisions

# +++++++++++++ object collisions ++++++++++++++
def bounds_check(co1, co2, fudge):
    """Returns True if object bounding boxes intersect.
    Have to add the fudge factor for collision margins"""
    check = False
    co1_max = None # will never return None if check is true
    co1_min = np.min(co1, axis=0)
    co2_max = np.max(co2, axis=0)

    if np.all(co2_max + fudge > co1_min):
        co1_max = np.max(co1, axis=0)
        co2_min = np.min(co2, axis=0)        
        
        if np.all(co1_max > co2_min - fudge):
            check = True

    return check, co1_min, co1_max # might as well reuse the checks


def triangle_bounds_check(tri_co, co_min, co_max, idxer, fudge):
    """Returns a bool aray indexing the triangles that
    intersect the bounds of the object"""

    # min check cull step 1
    tri_min = np.min(tri_co, axis=1) - fudge
    check_min = co_max > tri_min
    in_min = np.all(check_min, axis=1)
    
    # max check cull step 2
    idx = idxer[in_min]
    tri_max = np.max(tri_co[in_min], axis=1) + fudge
    check_max = tri_max > co_min
    in_max = np.all(check_max, axis=1)
    in_min[idx[~in_max]] = False
    
    return in_min, tri_min[in_min], tri_max[in_max] # can reuse the min and max


def tri_back_check(co, tri_min, tri_max, idxer, fudge):
    """Returns a bool aray indexing the vertices that
    intersect the bounds of the culled triangles"""

    # min check cull step 1
    tb_min = np.min(tri_min, axis=0) - fudge
    check_min = co > tb_min
    in_min = np.all(check_min, axis=1)
    idx = idxer[in_min]
    
    # max check cull step 2
    tb_max = np.max(tri_max, axis=0) + fudge
    check_max = co[in_min] < tb_max
    in_max = np.all(check_max, axis=1)        
    in_min[idx[~in_max]] = False    
    
    return in_min 


# -------------------------------------------------------
def zxy_grid(co_y, tymin, tymax, subs, c, t, c_peat, t_peat):
    # create linespace grid between bottom and top of tri z
    #subs = 7
    t_min = np.min(tymin)
    t_max = np.max(tymax)
    divs = np.linspace(t_min, t_max, num=subs, dtype=np.float32)            
    
    # figure out which triangles and which co are in each section
    co_bools = (co_y > divs[:-1][:, nax]) & (co_y < divs[1:][:, nax])
    tri_bools = (tymin < divs[1:][:, nax]) & (tymax > divs[:-1][:, nax])

    for i, j in zip(co_bools, tri_bools):
        if (np.sum(i) > 0) & (np.sum(j) > 0):
            c3 = c[i]
            t3 = t[j]
        
            c_peat.append(np.repeat(c3, t3.shape[0]))
            t_peat.append(np.tile(t3, c3.shape[0]))



def zx_grid(co_x, txmin, txmax, subs, c, t, c_peat, t_peat, co_y, tymin, tymax):
    # create linespace grid between bottom and top of tri z
    #subs = 7
    t_min = np.min(txmin)
    t_max = np.max(txmax)
    divs = np.linspace(t_min, t_max, num=subs, dtype=np.float32)            
    
    # figure out which triangles and which co are in each section
    co_bools = (co_x > divs[:-1][:, nax]) & (co_x < divs[1:][:, nax])
    tri_bools = (txmin < divs[1:][:, nax]) & (txmax > divs[:-1][:, nax])

    for i, j in zip(co_bools, tri_bools):
        if (np.sum(i) > 0) & (np.sum(j) > 0):
            c2 = c[i]
            t2 = t[j]
            
            zxy_grid(co_y[i], tymin[j], tymax[j], subs, c2, t2, c_peat, t_peat)


def z_grid(co_z, tzmin, tzmax, subs, co_x, txmin, txmax, co_y, tymin, tymax):
    # create linespace grid between bottom and top of tri z
    #subs = 7
    t_min = np.min(tzmin)
    t_max = np.max(tzmax)
    divs = np.linspace(t_min, t_max, num=subs, dtype=np.float32)
            
    # figure out which triangles and which co are in each section
    co_bools = (co_z > divs[:-1][:, nax]) & (co_z < divs[1:][:, nax])
    tri_bools = (tzmin < divs[1:][:, nax]) & (tzmax > divs[:-1][:, nax])

    c_ranger = np.arange(co_bools.shape[1])
    t_ranger = np.arange(tri_bools.shape[1])

    c_peat = []
    t_peat = []

    for i, j in zip(co_bools, tri_bools):
        if (np.sum(i) > 0) & (np.sum(j) > 0):
            c = c_ranger[i]
            t = t_ranger[j]

            zx_grid(co_x[i], txmin[j], txmax[j], subs, c, t, c_peat, t_peat, co_y[i], tymin[j], tymax[j])
    
    if (len(c_peat) == 0) | (len(t_peat) == 0):
        return None, None
    
    return np.hstack(c_peat), np.hstack(t_peat)
# -------------------------------------------------------

    
"""Combined with numexpr the first check min and max is faster
    Combined without numexpr is slower. It's better to separate min and max"""
def v_per_tri(co, tri_min, tri_max, idxer, tridexer, c_peat=None, t_peat=None):
    """Checks each point against the bounding box of each triangle"""

    co_x, co_y, co_z = co[:, 0], co[:, 1], co[:, 2]
    
    subs = 7
    #subs = bpy.data.objects['Plane.002'].modeling_cloth_grid_size
    
    c_peat, t_peat = z_grid(co_z, tri_min[:, 2], tri_max[:, 2], subs, co_x, tri_min[:, 0], tri_max[:, 0], co_y, tri_min[:, 1], tri_max[:, 1])
    
    if c_peat is None:
        return

    # X
    # Step 1 check x_min (because we're N squared here we break it into steps)
    check_x_min = co_x[c_peat] > tri_min[:, 0][t_peat]
    c_peat = c_peat[check_x_min]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_x_min]

    # Step 2 check x max
    check_x_max = co_x[c_peat] < tri_max[:, 0][t_peat]
    c_peat = c_peat[check_x_max]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_x_max]
    
    # Y
    # Step 3 check y min    
    check_y_min = co_y[c_peat] > tri_min[:, 1][t_peat]
    c_peat = c_peat[check_y_min]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_y_min]

    # Step 4 check y max
    check_y_max = co_y[c_peat] < tri_max[:, 1][t_peat]
    c_peat = c_peat[check_y_max]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_y_max]

    # Z
    # Step 5 check z min    
    check_z_min = co_z[c_peat] > tri_min[:, 2][t_peat]
    c_peat = c_peat[check_z_min]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_z_min]

    # Step 6 check y max
    check_z_max = co_z[c_peat] < tri_max[:, 2][t_peat]
    c_peat = c_peat[check_z_max]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_z_max]    

    return idxer[c_peat], t_peat
    #return c_peat, t_peat


def inside_triangles(tri_vecs, v2, co, tri_co_2, cidx, tidx, nor, ori, in_margin, offset=None):
    idxer = np.arange(in_margin.shape[0], dtype=np.int32)[in_margin]
    #return
    #r_co = co[cidx[in_margin]]    
    #r_tri = tri_co_2[tidx[in_margin]]
    
    v0 = tri_vecs[:,0]
    v1 = tri_vecs[:,1]
    
    d00_d11 = np.einsum('ijk,ijk->ij', tri_vecs, tri_vecs)
    d00 = d00_d11[:,0]
    d11 = d00_d11[:,1]
    d01 = np.einsum('ij,ij->i', v0, v1)
    d02 = np.einsum('ij,ij->i', v0, v2)
    d12 = np.einsum('ij,ij->i', v1, v2)

    div = 1 / (d00 * d11 - d01 * d01)
    u = (d11 * d02 - d01 * d12) * div
    v = (d00 * d12 - d01 * d02) * div
    
    # !!! Watch out for this number. It could affect speed !!! 
    if offset:
        check = (u > -offset) & (v > -offset) & (u + v < offset + 1)
    else:
        check = (u > 0) & (v > 0) & (u + v < 1)
    in_margin[idxer] = check


def object_collide(cloth, object):
    
    # for doing static cling
    #   cloth.col_idx = np.array([], dtype=np.int32)
    #   cloth.re_col = np.empty((0,3), dtype=np.float32)
    
    #proxy = object.ob.to_mesh(bpy.context.evaluated_depsgraph_get(), True, calc_undeformed=False)
    dg = object.dg
    
    #proxy = object.ob.to_mesh()
    proxy = object.ob.evaluated_get(dg).data
    
    
    proxy_in_place(object, proxy)
    apply_in_place(cloth.ob, cloth.co, cloth)

    inner_margin = object.ob.modeling_cloth_inner_margin
    outer_margin = object.ob.modeling_cloth_outer_margin
    fudge = max(inner_margin, outer_margin)

    # check object bounds: (need inner and out margins to adjust box size)
    box_check, co1_min, co1_max = bounds_check(cloth.co, object.co, fudge)
    # check for triangles inside the cloth bounds
    
    if box_check:

        proxy_v_normals_in_place(object, True, proxy)
        # coorect normals for collider scale:
        object.v_normals /= np.array(object.ob.scale)
        tri_co = object.co[object.tridex]
        tri_vo = object.vel[object.tridex]

        tris_in, tri_min, tri_max = triangle_bounds_check(tri_co, co1_min, co1_max, object.tridexer, fudge)#, object.ob.dimensions)

        # check for verts in the bounds around the culled triangles
        if np.any(tris_in):    
            tri_co_2 = tri_co[tris_in]
            back_check = tri_back_check(cloth.co, tri_min, tri_max, cloth.idxer, fudge)

            # begin every vertex co against every tri
            if np.any(back_check):
                v_tris = v_per_tri(cloth.co[back_check], tri_min, tri_max, cloth.idxer[back_check], object.tridexer[tris_in])
                
                if v_tris is not None:
                    # update the normals. cross_vecs used by barycentric tri check
                    #print(v_tris[0])
                    #print(v_tris[1])

                    # move the surface along the vertex normals by the outer margin distance
                    marginalized = (object.co + object.v_normals * outer_margin)[object.tridex]
                    tri_normals_in_place(object, marginalized)
                    
                    # add normals to make extruded tris
                    u_norms = object.normals[tris_in]
                                        
                    cidx, tidx = v_tris
                    ori = object.origins[tris_in][tidx]
                    nor = u_norms[tidx]
                    vec2 = cloth.co[cidx] - ori
                    
                    d = np.einsum('ij, ij->i', nor, vec2) # nor is unit norms
                    in_margin = (d > -(inner_margin + outer_margin)) & (d < 0)#outer_margin) (we have offset outer margin)
                    
                    # <<<--- Inside triangle check --->>>
                    # will overwrite in_margin:
                    cross_2 = object.cross_vecs[tris_in][tidx][in_margin]
                    inside_triangles(cross_2, vec2[in_margin], cloth.co, marginalized[tris_in], cidx, tidx, nor, ori, in_margin)
                    
                    if np.any(in_margin):
                        # collision response --------------------------->>>
                        t_in = tidx[in_margin]
                        
                        tri_vo = tri_vo[tris_in]
                        tri_vel1 = np.mean(tri_co_2[t_in], axis=1)
                        tri_vel2 = np.mean(tri_vo[t_in], axis=1)
                        tvel = tri_vel1 - tri_vel2

                        col_idx = cidx[in_margin] 
                        cloth.co[col_idx] -= nor[in_margin] * (d[in_margin])[:, nax]
                        cloth.vel[col_idx] = tvel
                        
                        # for doing static cling
                        #   cloth.re_col = np.copy(cloth.co[col_idx])                        
                        #   cloth.col_idx = col_idx
                        
    object.vel[:] = object.co    
    revert_in_place(cloth.ob, cloth.co)
    #bpy.data.meshes.remove(proxy)


# self collider =============================================
def self_collide(cloth):

    margin = cloth.ob.modeling_cloth_self_collision_margin

    tri_co = cloth.tri_co

    tri_min = np.min(tri_co, axis=1) - margin
    tri_max = np.max(tri_co, axis=1) + margin   

    # begin every vertex co against every tri
    v_tris = v_per_tri(cloth.co, tri_min, tri_max, cloth.idxer, cloth.tridexer)
    if v_tris is not None:
        cidx, tidx = v_tris

        u_norms = cloth.normals

        # don't check faces the verts are part of        
        check_neighbors = cidx[:, nax] == cloth.tridex[tidx]
        cull = np.any(check_neighbors, axis=1)
        cidx, tidx = cidx[~cull], tidx[~cull]
        
        ori = cloth.origins[tidx]
        nor = u_norms[tidx]
        vec2 = cloth.co[cidx] - ori
        
        d = np.einsum('ij, ij->i', nor, vec2) # nor is unit norms
        in_margin = (d > -margin) & (d < margin)
        # <<<--- Inside triangle check --->>>
        # will overwrite in_margin:
        cross_2 = cloth.cross_vecs[tidx][in_margin]
        inside_triangles(cross_2, vec2[in_margin], cloth.co, tri_co, cidx, tidx, nor, ori, in_margin, offset=0.0)
        
        if np.any(in_margin):
            # collision response --------------------------->>>
            t_in = tidx[in_margin]
            #tri_vel1 = np.mean(tri_co[t_in], axis=1)
            #tvel = np.mean(tri_vo[t_in], axis=1)
            #tvel = tri_vel1 - tri_vel2
            t_vel = np.mean(cloth.vel[cloth.tridex][t_in], axis=1)
            
            col_idx = cidx[in_margin] 
            d_in = d[in_margin]
    
            sign_margin = margin * np.sign(d_in) # which side of the face
            c_move = ((nor[in_margin] * d_in[:, nax]) - (nor[in_margin] * sign_margin[:, nax]))#) * -np.sign(d[in_margin])[:, nax]
            #c_move *= 1 / cloth.ob.modeling_cloth_grid_size
            #cloth.co[col_idx] -= ((nor[in_margin] * d_in[:, nax]) - (nor[in_margin] * sign_margin[:, nax]))#) * -np.sign(d[in_margin])[:, nax]
            cloth.co[col_idx] -= c_move * .7
            cloth.vel[col_idx] = 0
            cloth.vel[col_idx] = t_vel

    #object.vel[:] = object.co    
# self collider =============================================


class Collider(object):
    pass


class SelfCollider(object):
    pass


def create_collider():
    col = Collider()
    col.ob = bpy.context.active_object

    dg = bpy.context.evaluated_depsgraph_get()
    col.dg = dg
    # get proxy
    #proxy = col.ob.to_mesh(bpy.context.evaluated_depsgraph_get(), True, calc_undeformed=False)
    #proxy = col.ob.to_mesh()
    proxy = col.ob.evaluated_get(dg).data
    
    col.co = get_proxy_co(col.ob, None, proxy)
    col.idxer = np.arange(col.co.shape[0], dtype=np.int32)
    proxy_in_place(col, proxy)
    col.v_normals = proxy_v_normals(col.ob, proxy)
    col.vel = np.copy(col.co)
    col.tridex = triangulate(proxy)
    col.tridexer = np.arange(col.tridex.shape[0], dtype=np.int32)
    # cross_vecs used later by barycentric tri check
    proxy_v_normals_in_place(col, True, proxy)
    marginalized = col.co + col.v_normals * col.ob.modeling_cloth_outer_margin
    col.cross_vecs, col.origins, col.normals = get_tri_normals(marginalized[col.tridex])    
    
    # remove proxy
    #bpy.data.meshes.remove(proxy, do_unlink=True, do_id_user=True, do_ui_user=True)
    return col


# Self collision object
def create_self_collider():
    col = Collider()
    col.ob = bpy.context.active_object
    col.co = get_co(col.ob, None)
    proxy_in_place(col)
    col.v_normals = proxy_v_normals(col.ob)
    col.vel = np.copy(col.co)
    col.tridexer = np.arange(col.tridex.shape[0], dtype=np.int32)
    # cross_vecs used later by barycentric tri check
    proxy_v_normals_in_place(col)
    marginalized = col.co + col.v_normals * col.ob.modeling_cloth_outer_margin
    col.cross_vecs, col.origins, col.normals = get_tri_normals(marginalized[col.tridex])    

    return col


# collide object updater
def collision_object_update(self, context):
    """Updates the collider object"""    
    collide = self.modeling_cloth_object_collision
    # remove objects from dict if deleted
    cull_list = []
    if 'colliders' in extra_data:
        if extra_data['colliders'] is not None:   
            if not collide:
                if self.name in extra_data['colliders']:
                    del(extra_data['colliders'][self.name])
            for i in extra_data['colliders']:
                remove = True
                if i in bpy.data.objects:
                    if bpy.data.objects[i].type == "MESH":
                        if bpy.data.objects[i].modeling_cloth_object_collision:
                            remove = False
                if remove:
                    cull_list.append(i)
    for i in cull_list:
        del(extra_data['colliders'][i])

    # add class to dict if true.
    if collide:    
        if 'colliders' not in extra_data:    
            extra_data['colliders'] = {}
        if extra_data['colliders'] is None:
            extra_data['colliders'] = {}
        extra_data['colliders'][self.name] = create_collider()

    
# cloth object detect updater:
def cloth_object_update(self, context):
    """Updates the cloth object when detecting."""
    print("ran the detect updater. It did nothing.")


def manage_animation_handler(self, context):
    if self.modeling_cloth_handler_frame:
        self["modeling_cloth_handler_scene"] = False
        update_pin_group()
    
    if handler_frame in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(handler_frame)
    
    if len(data) > 0:
        bpy.app.handlers.frame_change_post.append(handler_frame)

        
def manage_continuous_handler(self, context):    
    if self.modeling_cloth_handler_scene:
        self["modeling_cloth_handler_frame"] = False
        update_pin_group()
    
    
    if bpy.app.timers.is_registered(mc_handler):
        bpy.app.timers.unregister(mc_handler)
    #if handler_scene in bpy.app.handlers.scene_update_post:
        #bpy.app.handlers.scene_update_post.remove(handler_scene)
    
    if len(data) > 0:
        bpy.app.timers.register(mc_handler)
        #bpy.app.handlers.scene_update_post.append(handler_scene)
    

# =================  Handler  ======================
def handler_frame(scene):

    items = data.items()
    if len(items) == 0:
        for i in bpy.data.objects:
            i.modeling_cloth = False
        
        for i in bpy.app.handlers.frame_change_post:
            if i.__name__ == 'handler_frame':
                bpy.app.handlers.frame_change_post.remove(i)

        if bpy.app.timers.is_registered(mc_handler):
            bpy.app.timers.unregister(mc_handler)        
        
                
        #for i in bpy.app.handlers.scene_update_post:
            #if i.__name__ == 'handler_scene':
                #bpy.app.handlers.scene_update_post.remove(i)                
    
    for i, cloth in items:    
        if i in bpy.data.objects: # using the name. The name could change
            if cloth.ob.modeling_cloth_handler_frame:    
                run_handler(cloth)
                if cloth.ob.modeling_cloth_auto_reset:
                    if bpy.context.scene.frame_current <= 1:    
                        reset_shapes(cloth.ob)
        else:
            del(data[i])
            break


def mc_handler():

    items = data.items()
    if len(items) == 0:
        for i in bpy.data.objects:
            i.modeling_cloth = False

        for i in bpy.app.handlers.frame_change_post:
            if i.__name__ == 'handler_frame':
                bpy.app.handlers.frame_change_post.remove(i)
        
        bpy.app.timers.unregister(mc_handler)
        #for i in bpy.app.handlers.scene_update_post:
            #if i.__name__ == 'handler_scene':
                #bpy.app.handlers.scene_update_post.remove(i)                
    
    for i, cloth in items:    
        if i in bpy.data.objects: # using the name. The name could change
            if cloth.ob.modeling_cloth_handler_scene:    
                run_handler(cloth)
        else:
            del(data[i])
            break

    return 0


def update_group_weight_falloff(self, context):
    ob = get_last_object()[1]
    fallof = ob.modeling_cloth_vertex_pin_fallof
    cloth = data[ob.name]
    cloth.blend_weights = cloth.pin_weights[cloth.weighted][:,nax] * fallof


def update_extend_springs(self, context):
    ob = get_last_object()[1]
    cloth = data[ob.name]
    if ob.modeling_cloth_extend_springs:    
        get_extend_springs(cloth, extend_springs=True)
        svecs = cloth.sco[cloth.eidx[:, 1]] - cloth.sco[cloth.eidx[:, 0]]
        cloth.sdots = np.einsum('ij,ij->i', svecs, svecs)        
        cloth.eidx_tiler = cloth.eidx.T.ravel()
        return

    get_extend_springs(cloth)
    svecs = cloth.sco[cloth.eidx[:, 1]] - cloth.sco[cloth.eidx[:, 0]]
    cloth.sdots = np.einsum('ij,ij->i', svecs, svecs)        
    cloth.eidx_tiler = cloth.eidx.T.ravel()
    

def update_softgoal_object(self, context):
    ob = get_last_object()[1]
    cloth = data[ob.name]


def pause_update(self, context):
    if not self.modeling_cloth_pause:
        update_pin_group()


def global_setup():
    global data, extra_data
    data = bpy.context.scene.modeling_cloth_data_set
    extra_data = bpy.context.scene.modeling_cloth_data_set_extra    
    extra_data['alert'] = False
    extra_data['drag_alert'] = False
    extra_data['clicked'] = False


def init_cloth(self, context):
    """!!!! This runs once for evey object in the scene !!!!"""
    global data, extra_data
    
    sce = bpy.context.scene
    if sce is None:
        print("scene was None !!!!!!!!!!!!!!!!!!!!!!")
        print("scene was None !!!!!!!!!!!!!!!!!!!!!!")
        print("scene was None !!!!!!!!!!!!!!!!!!!!!!")
        return
    
    data = sce.modeling_cloth_data_set
    extra_data = sce.modeling_cloth_data_set_extra
    extra_data['colliders'] = None
    
    extra_data['alert'] = False
    extra_data['drag_alert'] = False
    extra_data['last_object'] = self
    extra_data['clicked'] = False
    
    # object collisions
    colliders = [i for i in bpy.data.objects if i.modeling_cloth_object_collision]
    if len(colliders) == 0:    
        extra_data['colliders'] = None    
    
    # iterate through dict: for i, j in d.items()
    if self.modeling_cloth:
        cloth = create_instance() # generate an instance of the class
        data[cloth.name] = cloth  # store class in dictionary using the object name as a key
    
    remove = False    

    cull = [] # can't delete dict items while iterating
    
    # check if item is still in scene and not deleted by user.
    for i, value in data.items():
        if i not in bpy.data.objects:
            remove = True
            cull.append(i)
        
        if not remove:    
            if not value.ob.modeling_cloth:
                cull.append(i) # store keys to delete
        remove = False
        
    for i in cull:
        del data[i]


def main(context, event):
    # get the context arguments
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    coord = event.mouse_region_x, event.mouse_region_y

    # get the ray from the viewport and mouse
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
        ray_origin_obj = matrix_inv @ ray_origin
        ray_target_obj = matrix_inv @ ray_target
        ray_direction_obj = ray_target_obj - ray_origin_obj

        bv = mathutils.bvhtree.BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
        location, normal, face_index, success = bv.ray_cast(ray_origin_obj, view_vector)


        # cast the ray
        #success, location, normal, face_index = obj.ray_cast(ray_origin_obj, ray_direction_obj)

        if success:
            return location, normal, face_index
        else:
            return None, None, None

    # cast rays and find the closest object
    best_length_squared = -1.0
    best_obj = None
    for obj, matrix in visible_objects_and_duplis():

        m_state = []
        for m in obj.modifiers:
            m_state.append(m.show_viewport)
            m.show_viewport = False
        bpy.context.evaluated_depsgraph_get()
        
        hit, normal, face_index = obj_ray_cast(obj, matrix)
        
        for m, s in zip(obj.modifiers, m_state):
            m.show_viewport = s

        if hit is not None:
            hit_world = matrix @ hit
            vidx = [v for v in obj.data.polygons[face_index].vertices]
            verts = np.array([matrix @ obj.data.shape_keys.key_blocks['modeling cloth key'].data[v].co for v in obj.data.polygons[face_index].vertices])
            vecs = verts - np.array(hit_world)
            closest = vidx[np.argmin(np.einsum('ij,ij->i', vecs, vecs))]
            length_squared = (hit_world - ray_origin).length_squared
            if best_obj is None or length_squared < best_length_squared:
                best_length_squared = length_squared
                best_obj = obj
                guide.location = matrix @ obj.data.shape_keys.key_blocks['modeling cloth key'].data[closest].co
                extra_data['latest_hit'] = matrix @ obj.data.shape_keys.key_blocks['modeling cloth key'].data[closest].co
                extra_data['name'] = obj.name
                extra_data['obj'] = obj
                extra_data['closest'] = closest
                
                if extra_data['just_clicked']:
                    extra_data['just_clicked'] = False
                    best_length_squared = length_squared
                    best_obj = obj
                   

# sewing --------->>>
class ModelingClothSew(bpy.types.Operator):
    """For connected two edges with sew lines"""
    bl_idname = "object.modeling_cloth_create_sew_lines"
    bl_label = "Modeling Cloth Create Sew Lines"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        
        obj = bpy.context.active_object
        
        mode = obj.mode
        if mode != "EDIT":
            bpy.ops.object.mode_set(mode="EDIT")
        
        create_sew_edges()
        bpy.ops.object.mode_set(mode="EDIT")            

        return {'FINISHED'}
# sewing --------->>>


class ModelingClothPin(bpy.types.Operator):
    """Modal ray cast for placing pins"""
    bl_idname = "view3d.modeling_cloth_pin"
    bl_label = "Modeling Cloth Pin"
    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self):
        extra_data['just_clicked'] = False
        
    def modal(self, context, event):
        bpy.context.window.cursor_set("CROSSHAIR")
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'NUMPAD_0',
        'NUMPAD_PERIOD','NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4',
         'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9'}:
            # allow navigation
            return {'PASS_THROUGH'}
        elif event.type == 'MOUSEMOVE':
            main(context, event)
            return {'RUNNING_MODAL'}
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if extra_data['latest_hit'] is not None:
                if extra_data['name'] is not None:
                    closest = extra_data['closest']
                    name = extra_data['name']
                    e = bpy.data.objects.new('modeling_cloth_pin', None)
                    e.location = extra_data['latest_hit']
                    bpy.context.collection.objects.link(e)
                    bpy.context.evaluated_depsgraph_get()
                    #e.show_x_ray = True
                    e.select_set(True)# = True
                    #e.empty_draw_size = .1
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
                bpy.context.view_layer.objects.active = ob
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
#[â€˜DEFAULTâ€™, â€˜NONEâ€™, â€˜WAITâ€™, â€˜CROSSHAIRâ€™, â€˜MOVE_Xâ€™, â€˜MOVE_Yâ€™, â€˜KNIFEâ€™, â€˜TEXTâ€™, â€˜PAINT_BRUSHâ€™, â€˜HANDâ€™, â€˜SCROLL_Xâ€™, â€˜SCROLL_Yâ€™, â€˜SCROLL_XYâ€™, â€˜EYEDROPPERâ€™]

def main_drag(context, event):
    # get the context arguments
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    coord = event.mouse_region_x, event.mouse_region_y

    # get the ray from the viewport and mouse
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

    ray_target = ray_origin + view_vector

    def visible_objects_and_duplis():
        """Loop over (object, matrix) pairs (mesh only)"""

        for obj in context.visible_objects:
            if obj.type == 'MESH':
                if obj.modeling_cloth:    
                    # disable modifiers and store states
                    #mods = [i.show_viewport for i in obj.modifiers]
                    #for i in obj.modifiers:
                    #    i.show_viewport = False
                    yield (obj, obj.matrix_world.copy())#, mods)

    def obj_ray_cast(obj, matrix):
        """Wrapper for ray casting that moves the ray into object space"""

        # get the ray relative to the object
        matrix_inv = matrix.inverted()
        ray_origin_obj = matrix_inv @ ray_origin
        ray_target_obj = matrix_inv @ ray_target
        ray_direction_obj = ray_target_obj - ray_origin_obj

        bv = mathutils.bvhtree.BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
        location, normal, face_index, success = bv.ray_cast(ray_origin_obj, view_vector)
        
        if success:
            return location, normal, face_index, ray_target
        else:
            return None, None, None, ray_target

    # cast rays and find the closest object
    best_length_squared = -1.0
    best_obj = None
    
    for obj, matrix in visible_objects_and_duplis():
        
        m_state = []
        for m in obj.modifiers:
            m_state.append(m.show_viewport)
            m.show_viewport = False
        bpy.context.evaluated_depsgraph_get()
        
        hit, normal, face_index, target = obj_ray_cast(obj, matrix)
        extra_data['target'] = target
        
        for m, s in zip(obj.modifiers, m_state):
            m.show_viewport = s        
        
        if hit is not None:
                        
            hit_world = matrix @ hit
            length_squared = (hit_world - ray_origin).length_squared

            if best_obj is None or length_squared < best_length_squared:
                best_length_squared = length_squared
                best_obj = obj
                vidx = [v for v in obj.data.polygons[face_index].vertices]
                vert = obj.data.shape_keys.key_blocks['modeling cloth key'].data
            if best_obj is not None:    

                if extra_data['clicked']:    
                    extra_data['matrix'] = matrix.inverted()
                    data[best_obj.name].clicked = True
                    extra_data['stored_mouse'] = np.copy(target)
                    extra_data['vidx'] = vidx
                    extra_data['stored_vidx'] = np.array([vert[v].co for v in extra_data['vidx']])
                    extra_data['clicked'] = False

    if extra_data['stored_mouse'] is not None:
        move = np.array(extra_data['target']) - extra_data['stored_mouse']
        extra_data['move'] = (move @ np.array(extra_data['matrix'])[:3, :3].T)
                   
                   
# dragger===
class ModelingClothDrag(bpy.types.Operator):
    """Modal ray cast for dragging"""
    bl_idname = "view3d.modeling_cloth_drag"
    bl_label = "Modeling Cloth Drag"
    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self):
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
            main_drag(context, event)
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


class DeletePins(bpy.types.Operator):
    """Delete modeling cloth pins and clear pin list for current object"""
    bl_idname = "object.delete_modeling_cloth_pins"
    bl_label = "Delete Modeling Cloth Pins"
    bl_options = {'REGISTER', 'UNDO'}    
    def execute(self, context):

        ob = get_last_object() # returns tuple with list and last cloth objects or None
        if ob is not None:
            l_copy = data[ob[1].name].pin_list[:]
            h_copy = data[ob[1].name].hook_list[:]
            for i in range(len(data[ob[1].name].hook_list)):
                if data[ob[1].name].hook_list[i].select_get():
                    bpy.data.objects.remove(data[ob[1].name].hook_list[i])
                    l_copy.remove(data[ob[1].name].pin_list[i]) 
                    h_copy.remove(data[ob[1].name].hook_list[i]) 
            
            data[ob[1].name].pin_list = l_copy
            data[ob[1].name].hook_list = h_copy

        bpy.context.view_layer.objects.active = ob[1]
        return {'FINISHED'}


class SelectPins(bpy.types.Operator):
    """Select modeling cloth pins for current object"""
    bl_idname = "object.select_modeling_cloth_pins"
    bl_label = "Select Modeling Cloth Pins"
    bl_options = {'REGISTER', 'UNDO'}    
    def execute(self, context):
        ob = get_last_object() # returns list and last cloth objects or None
        if ob is not None:
            for i in data[ob[1].name].hook_list:
                i.select_set(True)

        return {'FINISHED'}


class PinSelected(bpy.types.Operator):
    """Add pins to verts selected in edit mode"""
    bl_idname = "object.modeling_cloth_pin_selected"
    bl_label = "Modeling Cloth Pin Selected"
    bl_options = {'REGISTER', 'UNDO'}    
    def execute(self, context):
        ob = bpy.context.active_object
        bpy.ops.object.mode_set(mode='OBJECT')
        sel = [i.index for i in ob.data.vertices if i.select]
                
        name = ob.name
        matrix = ob.matrix_world.copy()
        for v in sel:    
            e = bpy.data.objects.new('modeling_cloth_pin', None)
            bpy.context.collection.objects.link(e)
            if ob.active_shape_key is None:    
                closest = matrix @ ob.data.vertices[v].co# * matrix
            else:
                closest = matrix * ob.active_shape_key.data[v].co# * matrix
            e.location = closest #* matrix
            #e.show_x_ray = True
            e.select_set(True)
            #e.empty_draw_size = .1
            data[name].pin_list.append(v)
            data[name].hook_list.append(e)            
            ob.select_set(False)
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


class ApplyClothToMesh(bpy.types.Operator):
    """Apply cloth effects to mesh for export."""
    bl_idname = "object.modeling_cloth_apply_cloth_to_mesh"
    bl_label = "Modeling Cloth Remove Virtual Spring"
    bl_options = {'REGISTER', 'UNDO'}        
    def execute(self, context):
        ob = get_last_object()[1]
        v_count = len(ob.data.vertices)
        co = np.zeros(v_count * 3, dtype=np.float32)
        ob.data.shape_keys.key_blocks['modeling cloth key'].data.foreach_get('co', co)
        ob.data.shape_keys.key_blocks['Basis'].data.foreach_set('co', co)
        ob.data.shape_keys.key_blocks['Basis'].mute = True
        ob.data.shape_keys.key_blocks['Basis'].mute = False
        ob.data.vertices.foreach_set('co', co)
        ob.data.update()

        return {'FINISHED'}


def create_properties():            

    bpy.types.Object.modeling_cloth = bpy.props.BoolProperty(name="Modeling Cloth", 
        description="For toggling modeling cloth", 
        default=False, update=init_cloth)

    bpy.types.Object.modeling_cloth_dynamic_source = bpy.props.BoolProperty(name="Modeling Cloth Dynamic Source", 
        description="If the source springs change during animation", 
        default=False)

    bpy.types.Object.modeling_cloth_floor = bpy.props.BoolProperty(name="Modeling Cloth Floor", 
        description="Stop at floor", 
        default=False)

    bpy.types.Object.modeling_cloth_pause = bpy.props.BoolProperty(name="Modeling Cloth Pause", 
        description="Stop without removing data",
        default=True, update=pause_update)
    
    # handler type ----->>>        
    bpy.types.Object.modeling_cloth_handler_scene = bpy.props.BoolProperty(name="Modeling Cloth Continuous Update", 
        description="Choose continuous update", 
        default=False, update=manage_continuous_handler)        

    bpy.types.Object.modeling_cloth_handler_frame = bpy.props.BoolProperty(name="Modeling Cloth Handler Animation Update", 
        description="Choose animation update", 
        default=False, update=manage_animation_handler)
        
    bpy.types.Object.modeling_cloth_auto_reset = bpy.props.BoolProperty(name="Modeling Cloth Reset at Frame 1", 
        description="Automatically reset if the current frame number is 1 or less", 
        default=False)#, update=manage_handlers)        
    # ------------------>>>

    bpy.types.Object.modeling_cloth_noise = bpy.props.FloatProperty(name="Modeling Cloth Noise", 
        description="Set the noise strength", 
        default=0.001, precision=4, min=0, max=1, update=refresh_noise)

    bpy.types.Object.modeling_cloth_noise_decay = bpy.props.FloatProperty(name="Modeling Cloth Noise Decay", 
        description="Multiply the noise by this value each iteration", 
        default=0.99, precision=4, min=0, max=1)#, update=refresh_noise_decay)

    # spring forces ------------>>>
    bpy.types.Object.modeling_cloth_spring_force = bpy.props.FloatProperty(name="Modeling Cloth Spring Force", 
        description="Set the spring force", 
        default=1.0, precision=4, min=0, max=2.5)#, update=refresh_noise)

    bpy.types.Object.modeling_cloth_push_springs = bpy.props.FloatProperty(name="Modeling Cloth Push Spring Force", 
        description="Set the push spring force", 
        default=1.0, precision=4, min=0, max=2.5)#, update=refresh_noise)
    
    # bend springs
    bpy.types.Object.modeling_cloth_bend_stiff = bpy.props.FloatProperty(name="Modeling Cloth Bend Spring Force", 
        description="Set the bend spring force", 
        default=0.0, precision=4, min=0, max=10, soft_max=1)#, update=refresh_noise)
    # -------------------------->>>

    bpy.types.Object.modeling_cloth_gravity = bpy.props.FloatProperty(name="Modeling Cloth Gravity", 
        description="Modeling cloth gravity", 
        default=0.0, precision=4, soft_min=-10, soft_max=10, min=-1000, max=1000)

    bpy.types.Object.modeling_cloth_iterations = bpy.props.IntProperty(name="Stiffness", 
        description="How stiff the cloth is", 
        default=2, min=1, max=500)#, update=refresh_noise_decay)

    bpy.types.Object.modeling_cloth_velocity = bpy.props.FloatProperty(name="Velocity", 
        description="Cloth keeps moving", 
        default=.98, min= -200, max=200, soft_min= -1, soft_max=1)#, update=refresh_noise_decay)

    # Wind. Note, wind should be measured against normal and be at zero when normals are at zero. Squared should work
    bpy.types.Object.modeling_cloth_wind_x = bpy.props.FloatProperty(name="Wind X", 
        description="Not the window cleaner", 
        default=0, min= -10, max=10, soft_min= -1, soft_max=1)#, update=refresh_noise_decay)

    bpy.types.Object.modeling_cloth_wind_y = bpy.props.FloatProperty(name="Wind Y", 
        description="Y? Because wind is cool", 
        default=0, min= -10, max=10, soft_min= -1, soft_max=1)#, update=refresh_noise_decay)

    bpy.types.Object.modeling_cloth_wind_z = bpy.props.FloatProperty(name="Wind Z", 
        description="It's windzee outzide", 
        default=0, min= -10, max=10, soft_min= -1, soft_max=1)#, update=refresh_noise_decay)

    bpy.types.Object.modeling_cloth_turbulence = bpy.props.FloatProperty(name="Wind Turbulence", 
        description="Add Randomness to wind", 
        default=0, min=0, max=10, soft_min= 0, soft_max=1)#, update=refresh_noise_decay)

    # self collision ----->>>
    bpy.types.Object.modeling_cloth_self_collision = bpy.props.BoolProperty(name="Modeling Cloth Self Collsion", 
        description="Toggle self collision", 
        default=False, update=collision_data_update)

    bpy.types.Object.modeling_cloth_self_collision_margin = bpy.props.FloatProperty(name="Margin", 
        description="Self colide faces margin", 
        default=.08, precision=4, min= -1, max=1, soft_min= 0, soft_max=1)

    # extras ------->>>
    bpy.types.Object.modeling_cloth_inflate = bpy.props.FloatProperty(name="inflate", 
        description="add force to vertex normals", 
        default=0, precision=4, min= -10, max=10, soft_min= -1, soft_max=1)

    bpy.types.Object.modeling_cloth_sew = bpy.props.FloatProperty(name="sew", 
        description="add force to vertex normals", 
        default=0, precision=4, min= -10, max=10, soft_min= -1, soft_max=1)

    bpy.types.Object.modeling_cloth_extend_springs = bpy.props.BoolProperty(name="Modeling Cloth Softbody Goal", 
        description="Use this object as softbody goal", 
        default=False, update=update_extend_springs)

    bpy.types.Object.modeling_cloth_vertex_pin_fallof = bpy.props.FloatProperty(name="Vertex Pin Fallof", 
        description="Adjust strength of vertex weight pins", 
        default=1, precision=4, min= -10, max=10, soft_min= 0, soft_max=1)

    # softgoal objects ---------->>>
    bpy.types.Object.modeling_cloth_softbody_goal = bpy.props.BoolProperty(name="Modeling Cloth Softbody Goal", 
        description="Use this object as softbody goal", 
        default=False, update=update_softgoal_object)
    
    bpy.types.Object.modeling_cloth_softgoal_strength = bpy.props.FloatProperty(name="Softgoal Strength",
        description="strength of softgoal object", 
        default=0.1, precision=4, min= -10, max=10, soft_min= 0, soft_max=1)    

    bpy.types.Object.modeling_cloth_softbody_fixed_goal = bpy.props.BoolProperty(name="Modeling Cloth Softbody Fixed Goal", 
        description="Object Moves With Cloth", 
        default=False, update=update_softgoal_object)    
    
    # -------------->>>

    # external collisions ------->>>
    bpy.types.Object.modeling_cloth_object_collision = bpy.props.BoolProperty(name="Modeling Cloth Self Collsion", 
        description="Detect and collide with this object", 
        default=False, update=collision_object_update)

    bpy.types.Object.modeling_cloth_object_detect = bpy.props.BoolProperty(name="Modeling Cloth Self Collsion", 
        description="Detect collision objects", 
        default=True, update=cloth_object_update)    

    bpy.types.Object.modeling_cloth_outer_margin = bpy.props.FloatProperty(name="Modeling Cloth Outer Margin", 
        description="Collision margin on positive normal side of face", 
        default=0.04, precision=4, min=0, max=100, soft_min=0, soft_max=1000)
        
    bpy.types.Object.modeling_cloth_inner_margin = bpy.props.FloatProperty(name="Modeling Cloth Inner Margin", 
        description="Collision margin on negative normal side of face", 
        default=0.08, precision=4, min=0, max=100, soft_min=0, soft_max=1000)        
    # ---------------------------->>>
    
    # more collision stuff ------->>>
    bpy.types.Object.modeling_cloth_grid_size = bpy.props.IntProperty(name="Modeling Cloth Grid Size", 
    description="Max subdivisions for the dynamic broad phase grid", 
    default=10, min=0, max=1000, soft_min=0, soft_max=1000)
    
    # property dictionaries
    if "modeling_cloth_data_set" not in dir(bpy.types.Scene):
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

    # data storage
    del(bpy.types.Scene.modeling_cloth_data_set)
    del(bpy.types.Scene.modeling_cloth_data_set_extra)


class ModelingClothPanel(bpy.types.Panel):
    """Modeling Cloth Panel"""
    bl_label = "Modeling Cloth Panel"
    bl_idname = "Modeling Cloth"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"
    #gt_show = True
    
    def draw(self, context):
        status = False
        layout = self.layout

        ob = bpy.context.active_object

        col = layout.column(align=True)
        col.label(text='Support Addons')
        col.operator("object.modeling_cloth_donate", text="Donate")
        col.operator("object.modeling_cloth_collision_series", text="Buy Books")
        #col.operator("object.modeling_cloth_collision_series_kindle", text="Kindle")
        
        # tools
        col = layout.column(align=True)
        col.label(text="Tools")        
        col.operator("object.modeling_cloth_create_sew_lines", text="Sew Lines", icon="MOD_UVPROJECT")
        col.operator("object.modeling_cloth_apply_cloth_to_mesh", text="Apply to Mesh", icon="FILE_TICK")

        # softgoals
        col = layout.column(align=True)
        col.label(text="Soft Goals")

        if ob is not None:
            col.prop(ob ,"modeling_cloth_softbody_goal", text="Soft Goal", icon='MOD_SOFT')
            if ob.modeling_cloth_softbody_goal:
                col.prop(ob ,"modeling_cloth_softbody_fixed_goal", text="Fixed Goal", icon='PINNED')
                col.prop(ob ,"modeling_cloth_softgoal_strength", text="Goal Force", icon='OUTLINER_OB_FORCE_FIELD')
        
        # modeling cloth
        col = layout.column(align=True)
        col.label(text="Modeling Cloth")
        
        cloths = [i for i in bpy.data.objects if i.modeling_cloth] # so we can select an empty and keep the settings menu up
        if len(cloths) > 0:
            extra_data = bpy.context.scene.modeling_cloth_data_set_extra
            if 'alert' not in extra_data:
                global_setup()    
            status = extra_data['alert']
            if ob is not None:
                if ob.type != 'MESH' or status:
                    ob = extra_data['last_object']

        if ob is not None:
            if ob.type == 'MESH':
                col.prop(ob ,"modeling_cloth", text="Modeling Cloth", icon='SURFACE_DATA')               
                if ob.modeling_cloth:    
                    col.prop(ob ,"modeling_cloth_dynamic_source", text="Dynamic Source", icon='MOD_PHYSICS')
                    col = layout.column(align=True)                    
                    col.prop(ob ,"modeling_cloth_self_collision", text="Self Collision", icon='PHYSICS')
                    col.prop(ob ,"modeling_cloth_self_collision_margin", text="Self Margin")#, icon='PLAY')
                    
                pause = 'PAUSE'
                if ob.modeling_cloth_pause:
                    pause = 'PLAY'
                
                col.prop(ob ,"modeling_cloth_object_collision", text="Collider", icon="STYLUS_PRESSURE")
                if ob.modeling_cloth_object_collision:    
                    col.prop(ob ,"modeling_cloth_outer_margin", text="Outer Margin", icon="FORCE_FORCE")
                    col.prop(ob ,"modeling_cloth_inner_margin", text="Inner Margin", icon="STICKY_UVS_LOC")
                    col = layout.column(align=True)
                    
                col.label(text="Collide List:")
                colliders = [i.name for i in bpy.data.objects if i.modeling_cloth_object_collision]
                for i in colliders:
                    col.label(text=i)

                if ob.modeling_cloth:

                    # object collisions
                    col = layout.column(align=True)
                    col.label(text="Collisions")
                    if ob.modeling_cloth:    
                        col.prop(ob ,"modeling_cloth_object_detect", text="Object Collisions", icon="PHYSICS")

                    col = layout.column(align=True)
                    col.scale_y = 2.0

                    col = layout.column(align=True)
                    col.scale_y = 1.4
                    col.prop(ob, "modeling_cloth_grid_size", text="Grid Boxes", icon="MESH_GRID")
                    col.prop(ob, "modeling_cloth_handler_frame", text="Animation Update", icon="TRIA_RIGHT")
                    if ob.modeling_cloth_handler_frame:    
                        col.prop(ob, "modeling_cloth_auto_reset", text="Frame 1 Reset")
                    col.prop(ob, "modeling_cloth_handler_scene", text="Continuous Update", icon="TIME")
                    col = layout.column(align=True)
                    col.scale_y = 2.0
                    col.operator("object.modeling_cloth_reset", text="Reset")
                    col.alert = extra_data['drag_alert']
                    col.operator("view3d.modeling_cloth_drag", text="Grab")
                    col = layout.column(align=True)
                        
                    col.prop(ob ,"modeling_cloth_iterations", text="Iterations")#, icon='OUTLINER_OB_LATTICE')               
                    col.prop(ob ,"modeling_cloth_spring_force", text="Stiffness")#, icon='OUTLINER_OB_LATTICE')               
                    col.prop(ob ,"modeling_cloth_push_springs", text="Push Springs")#, icon='OUTLINER_OB_LATTICE')               
                    col.prop(ob ,"modeling_cloth_bend_stiff", text="Bend Springs")#, icon='CURVE_NCURVE')               
                    col.prop(ob ,"modeling_cloth_extend_springs", text="Extend Springs", icon='OUTLINER_OB_LATTICE')               
                    col.prop(ob ,"modeling_cloth_vertex_pin_fallof", text="Group Pin Fallof")#, icon='PLAY')               
                    col.prop(ob ,"modeling_cloth_noise", text="Noise")#, icon='PLAY')               
                    col.prop(ob ,"modeling_cloth_noise_decay", text="Decay Noise")#, icon='PLAY')               
                    col.prop(ob ,"modeling_cloth_gravity", text="Gravity")#, icon='PLAY')        
                    col.prop(ob ,"modeling_cloth_inflate", text="Inflate")#, icon='PLAY')        
                    col.prop(ob ,"modeling_cloth_sew", text="Sew Force")#, icon='PLAY')        
                    col.prop(ob ,"modeling_cloth_velocity", text="Velocity")#, icon='PLAY')        
                    col = layout.column(align=True)
                    col.label(text="Wind")                
                    col.prop(ob ,"modeling_cloth_wind_x", text="Wind X")#, icon='PLAY')        
                    col.prop(ob ,"modeling_cloth_wind_y", text="Wind Y")#, icon='PLAY')        
                    col.prop(ob ,"modeling_cloth_wind_z", text="Wind Z")#, icon='PLAY')        
                    col.prop(ob ,"modeling_cloth_turbulence", text="Turbulence")#, icon='PLAY')        
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
         
                # =============================
                col = layout.column(align=True)
                col.label(text='Support Addons')
                #col.operator("object.modeling_cloth_collision_series_kindle", text="Kindle")
                col.operator("object.modeling_cloth_donate", text="Donate")
                col.operator("object.modeling_cloth_collision_series", text="Buy Books")


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
        webbrowser.open("https://www.amazon.com/s?i=digital-text&rh=p_27%3ARich+Colburn&s=relevancerank&text=Rich+Colburn&ref=dp_byline_sr_ebooks_1")
        #imp.reload(webbrowser)
        #webbrowser.open("https://www.createspace.com/7164863")
        return
    if kindle:
        webbrowser.open("https://www.amazon.com/s?i=digital-text&rh=p_27%3ARich+Colburn&s=relevancerank&text=Rich+Colburn&ref=dp_byline_sr_ebooks_1")
        #imp.reload(webbrowser)
        #webbrowser.open("https://www.amazon.com/Formulacrum-Collision-Book-Rich-Colburn-ebook/dp/B0711P744G")
        return
    webbrowser.open("https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=4T4WNFQXGS99A")

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
    bpy.utils.register_class(ModelingClothSew)
    bpy.utils.register_class(ApplyClothToMesh)
    
    
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
    bpy.utils.unregister_class(ModelingClothSew)
    bpy.utils.unregister_class(ApplyClothToMesh)
    
    
    bpy.utils.unregister_class(CollisionSeries)
    bpy.utils.unregister_class(CollisionSeriesKindle)
    bpy.utils.unregister_class(Donate)
    
    
if __name__ == "__main__":
    register()

    # testing!!!!!!!!!!!!!!!!
    #generate_collision_data(bpy.context.active_object)
    # testing!!!!!!!!!!!!!!!!
    
    for i in bpy.data.objects:
        i.modeling_cloth = False
        i.modeling_cloth_object_collision = False
        
    for i in bpy.app.handlers.frame_change_post:
        if i.__name__ == 'handler_frame':
            bpy.app.handlers.frame_change_post.remove(i)
            
    if bpy.app.timers.is_registered(mc_handler):
        bpy.app.timers.unregister(mc_handler)

    
    #for i in bpy.app.handlers.scene_update_post:
        #if i.__name__ == 'handler_scene':
            #bpy.app.handlers.scene_update_post.remove(i)            
