# You are at the top. If you attempt to go any higher
#   you will go beyond the known limits of the code
#   universe where there are most certainly monsters

# to do:
#  check speed difference using 32bit float instead of default
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
#  When growing the source, the size of collision margins doesn's scale
#  add inflate


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


you_have_a_sense_of_humor = False
#you_have_a_sense_of_humor = True
if you_have_a_sense_of_humor:
    import antigravity


def closest_point_edge(e1, e2, p):
    '''Returns the location of the point on the edge'''
    vec1 = e2 - e1
    vec2 = p - e1
    d = np.dot(vec2, vec1) / np.dot(vec1, vec1)
    cp = e1 + vec1 * d 
    return cp


#testob = bpy.data.objects['P']
#eidx = np.zeros(len(testob.data.edges) * 2)
#testob.data.edges.foreach_get('vertices', eidx)
#p_count = len(testob.data.polygons)
#v_count = len(testob.data.vertices)
#co = np.zeros(v_count * 3)
#testob.data.vertices.foreach_get('co', co.ravel())
#co.shape = (v_count, 3)
#p_verts = [[i for i in p.vertices] for p in testob.data.polygons]
#centers = [np.mean(co[p_verts[i]], axis = 0) for i in range(p_count)]
#p_size = []

#for i in range(p_count):
#    vecs = co[p_verts[i]] - co[np.roll(p_verts[i], 1)]
#    vp_count = len(p_verts[i])
#    ccp = []
#    for c in range(vp_count):
#        cp = closest_point_edge(co[p_verts[i][c]], co[np.roll(p_verts[i], 1)[c]], centers[i])
#        ccp.append(cp)
#    dif = centers[i] - ccp
#    size = np.min(np.sqrt((np.einsum('ij,ij->i', dif, dif))))
#    p_size.append(size * .7)


def generate_collision_data(ob):
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
    # since both sids are being check it's unlikely that both sides could ever line up and pass through the gap
    obm = bmesh.new()
    obm.from_mesh(ob.data)


    obm.faces.ensure_lookup_table()
    p_count = len(obm.faces)
    v_count = len(obm.verts)
    
    per_face_v =  [[v.co for v in f.verts] for f in obm.faces]
    means = np.array([np.mean([v.co for v in f.verts], axis=0) for f in obm.faces], dtype=np.float32)
    
    # get sqared distance to closest vert in each face. (this will still work if the mesh isn't flat)
    sq_dist = []
    for i in range(p_count):
        dif = np.array(per_face_v[i]) - means[i]
        sq_dist.append(np.min(np.einsum('ij,ij->i', dif, dif)))
    
    # neighbors for excluding point face collisions.
    neighbors = np.tile(np.ones(p_count, dtype=np.bool), (v_count, 1))
    p_neighbors = [[f.index for f in v.link_faces] for v in obm.verts]
    #print(p_neighbors)
    for x in range(neighbors.shape[0]):
        neighbors[x][p_neighbors[x]] = False

    #count = 0
    #for x in np.nditer(neighbors, flags=['external_loop', 'buffered'], op_flags=['readwrite']):
        #count += 1
        #print(count)
    #for i in range(len(neighbors)):
    
    
        
    #print(neighbors)

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


def tile_eidx():
    '''currently not used. For option of calculating edges only once
    instead of duplicated in solver'''
    obm = get_bmesh()
    obm.verts.ensure_lookup_table()
    
    unique = []
    tile = []
    flip = []
        
    return unique, tile, flip


def get_bmesh(ob=None):
    ob = get_last_object()[1]
    obm = bmesh.new()
    if ob.mode == 'OBJECT':
        obm.from_mesh(ob.data)
    elif ob.mode == 'EDIT':
        obm = bmesh.from_edit_mesh(ob.data)
    return obm


def connected_by_polygons():
    obm = get_bmesh()
    obm.verts.ensure_lookup_table()
    virtual_edges = []
    for v in obm.verts:
        lv = np.hstack([[fv.index for fv in f.verts if fv != v] for f in v.link_faces])
        for i in np.unique(lv):
            virtual_edges.append([v.index, i])
    
    return np.array(virtual_edges, dtype=np.int32) # note there are many duplicates: [0,72], [72,0]


def get_last_object():
    """Finds cloth objects for keeping settings active
    while selecting other objects like pins"""
    cloths = [i for i in bpy.data.objects if i.modeling_cloth] # so we can select an empty and keep the settings menu up
    if bpy.context.object.modeling_cloth:
        return cloths, bpy.context.object
    
    if len(cloths) > 0:
        ob = extra_data['last_object']
        return cloths, ob
    return None


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
        bpy.data.meshes.remove(bpy.data.meshes['ModelingClothPinGuide'])
    

def scale_source(multiplier):
    """grow or shrink the source shape"""
    ob = get_last_object()[1]
    if ob is not None:
        if ob.modeling_cloth:
            count = len(ob.data.vertices)
            co = np.zeros(count*3, dtype=np.float32)
            ob.data.shape_keys.key_blocks['cloth source key'].data.foreach_get('co', co)
            co.shape = (count, 3)
            mean = np.mean(co, axis=0)
            co -= mean
            co *= multiplier
            co += mean
            ob.data.shape_keys.key_blocks['cloth source key'].data.foreach_set('co', co.ravel())                


def reset_shapes():
    """Sets the cloth key to match the source key.
    Will regenerate shape keys if they are missing"""
    if bpy.context.object.modeling_cloth:
        ob = bpy.context.object
    else:    
        ob = extra_data['last_object']

    if ob.data.shape_keys == None:
        ob.shape_key_add('Basis')    
    if 'cloth source key' not in ob.data.shape_keys.key_blocks:
        ob.shape_key_add('cloth source key')        
    if 'cloth key' not in ob.data.shape_keys.key_blocks:
        ob.shape_key_add('cloth key')        
        ob.data.shape_keys.key_blocks['cloth key'].value=1
    
    keys = ob.data.shape_keys.key_blocks
    count = len(ob.data.vertices)
    co = np.zeros(count * 3, dtype=np.float32)
    keys['cloth source key'].data.foreach_get('co', co)
    keys['cloth key'].data.foreach_set('co', co)

    data[ob.name].vel *= 0
    
    ob.data.shape_keys.key_blocks['cloth key'].mute = True
    ob.data.shape_keys.key_blocks['cloth key'].mute = False


def update_pin_group():
    """Updates the cloth data after changing mesh or vertex weight pins"""
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
    if 'cloth source key' not in cloth.ob.data.shape_keys.key_blocks:
        cloth.ob.shape_key_add('cloth source key')        
    if 'cloth key' not in cloth.ob.data.shape_keys.key_blocks:
        cloth.ob.shape_key_add('cloth key')        
        cloth.ob.data.shape_keys.key_blocks['cloth key'].value=1
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

    cloth.eidx = connected_by_polygons()
    eidx1 = connected_by_polygons()
    pindexer = np.arange(cloth.count, dtype=np.int32)[cloth.pin_bool]
    unpinned = np.in1d(eidx1[:, 0], pindexer)
    cloth.eidx = eidx1[unpinned]    
    cloth.pcount = pindexer.shape[0]
    cloth.pindexer = pindexer
    
    cloth.sco = np.zeros(cloth.count * 3, dtype=np.float32)
    cloth.ob.data.shape_keys.key_blocks['cloth source key'].data.foreach_get('co', cloth.sco)
    cloth.sco.shape = (cloth.count, 3)
    cloth.co = np.zeros(cloth.count * 3, dtype=np.float32)
    cloth.ob.data.shape_keys.key_blocks['cloth key'].data.foreach_get('co', cloth.co)
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
    
    cloth.l = np.unique(cloth.eidx[:, 0], return_inverse=True, return_counts=True)
    cloth.mix = (1/cloth.l[2][cloth.l[1]])[:, nax].astype(np.float32) # force gets divided by number of springs
    cloth.l_eidx = cloth.eidx[:, 0]

    if new:
        cloth.pin_list = []
        cloth.hook_list = []
    
    # collision======:
    # collision======:

    #setup for means with random numbers of v_counts in faces:-----
    cloth.p_count = len(cloth.ob.data.polygons)
    cloth.v_per_f = [[i for i in p.vertices] for p in cloth.ob.data.polygons]
    cloth.v_per_p = np.copy([[i for i in p.vertices] for p in cloth.ob.data.polygons])
    
    cloth.p_means_add_tiler = np.hstack(cloth.v_per_f)
    cloth.p_means = np.zeros(cloth.p_count * 3, dtype=np.float32)
    cloth.p_means.shape = (cloth.p_count, 3)
    cloth.v_count_per_f = np.array([len(p.vertices) for p in cloth.ob.data.polygons])
    cloth.tiler = np.hstack([[i for j in range(cloth.v_count_per_f[i])] for i in range(len(cloth.v_count_per_f))])
    cloth.poly_meaner = (1/cloth.v_count_per_f)[:, nax].astype(np.float32)
    np.add.at(cloth.p_means, cloth.tiler, cloth.co[cloth.p_means_add_tiler])
    cloth.p_means *= cloth.poly_meaner
    
    # could put in a check in case int 32 isn't big enough...
    cloth.cy_dists, cloth.point_mean_neighbors = generate_collision_data(cloth.ob)
    cloth.cy_dists *= cloth.ob.modeling_cloth_self_collision_cy_size
    
    nei = cloth.point_mean_neighbors.ravel() # eliminate neighbors for point in face check
    print('new=========================')
    cloth.v_repeater = np.repeat(np.arange(cloth.count, dtype=np.int32), cloth.p_count)[nei]
    #cloth.v_repeater = np.repeat(pindexer, cloth.p_count)[nei]
    cloth.p_repeater = np.tile(np.arange(cloth.p_count, dtype=np.int32),(cloth.count,))[nei]
    cloth.bool_repeater = np.ones(cloth.p_repeater.shape[0], dtype=np.bool)

    
    
    
    cloth.mean_idxer = np.arange(cloth.p_count)
    cloth.mean_tidxer = np.tile(cloth.mean_idxer, (cloth.count, 1))
    

    
    #setup for normals -----
    cloth.p_normals = np.zeros(cloth.p_count * 3, dtype=np.float32) # can get from shape key
    normals_idx = []
    for i in range(cloth.p_count):
        vecs = cloth.p_means[i] - cloth.co[cloth.v_per_f[i]]
        p1 = cloth.v_per_f[i][np.argmax(np.einsum('ij,ij->i', vecs, vecs))]
        cloth.v_per_f[i].remove(p1)
        vecs2 = co[p1] - cloth.co[cloth.v_per_f[i]]
        p2 = cloth.v_per_f[i][np.argmax(np.einsum('ij,ij->i', vecs2, vecs2))]
        e_mean = np.mean(co[[p1, p2]], axis=0)
        
        
        vecs3 = e_mean - cloth.co[cloth.v_per_f[i]]
        p3 = cloth.v_per_f[i][np.argmin(np.abs(np.einsum('j,ij->i', co[p1] - co[p2], vecs3)))]
        normals_idx.append([p1, p2, p3])
    
    cloth.normals_idx = np.array(normals_idx)
    cloth.norm_base = cloth.normals_idx[:,2]
    cloth.norm_pairs = cloth.normals_idx[:,:2]
    
    # I'm checking the centers against the other face centers so I guess I need
    #   the face means every frame.
    
    # I'll need to get the object transforms if I check for collisions on other objects



    #ees = [i for i in bpy.data.objects if i.name.startswith('ee')]
    #for i in range(len(cloth.center_vel_start)):
        #ees[i].location = cloth.center_vel_start[i]

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
        
        eidx = cloth.eidx
        l_eidx = cloth.l_eidx
        # can get a speedup here by indexing the edges and flipping instead of doing
        #   calculations on the tiled edges. (tile after doing math)
        cloth.ob.data.shape_keys.key_blocks['cloth source key'].data.foreach_get('co', cloth.sco.ravel())
        sco = cloth.sco
        sco.shape = (cloth.count, 3)
        cloth.ob.data.shape_keys.key_blocks['cloth key'].data.foreach_get('co', cloth.co.ravel())
        co = cloth.co
        co.shape = (cloth.count, 3)

        svecs = sco[eidx[:, 1]] - sco[eidx[:, 0]]
        sdots = np.einsum('ij,ij->i', svecs, svecs)

        co[cloth.pindexer] += cloth.noise[cloth.pindexer]
        cloth.noise *= cloth.ob.modeling_cloth_noise_decay
        #co[cloth.pindexer] -= cloth.vel[cloth.pindexer] 
        cloth.vel_start[:] = co
        force = cloth.ob.modeling_cloth_spring_force
        mix = cloth.mix * force
        for x in range(cloth.ob.modeling_cloth_iterations):    
            if force < 0.17:
                force += .01

            vecs = co[eidx[:, 1]] - co[eidx[:, 0]]
            dots = np.einsum('ij,ij->i', vecs, vecs)
            div = np.nan_to_num(sdots / dots)
            swap = vecs * np.sqrt(div)[:, nax]
            move = vecs - swap

            #---
            move *= mix # for stability: force multiplied by 1/number of springs
            #---
            
            np.add.at(cloth.co, l_eidx, move)    
            
            if len(cloth.pin_list) > 0:
                hook_co = np.array([cloth.ob.matrix_world.inverted() * i.matrix_world.to_translation() for i in cloth.hook_list])
                cloth.co[cloth.pin_list] = hook_co



        # floor ---
        if cloth.ob.modeling_cloth_floor:    
            floored = cloth.co[:,2] < 0        
            cloth.vel[:,2][floored] *= -1
            cloth.vel[floored] *= .1
            cloth.co[:, 2][floored] = 0
        # floor ---

        #target a mesh======================:
        #target a mesh======================:
            
            #use the v_normal drop or possible a ne closest point on mesh method
            #   as a force to move the cloth towards another object using a distance off
            #   the mesh as a target.
        
        
        #target a mesh======================:
        #target a mesh======================:        


        vel_dif = cloth.vel_start - cloth.co

        cloth.vel += vel_dif
        

        #collision=====================================
        #collision=====================================
        # for grow and shrink, the distance will need to change
        #   it gets recaluclated when going in and out of edit mode already...
        #self_collision = False
        self_col = cloth.ob.modeling_cloth_self_collision

        if self_col:
            V3 = [] # because I'm multiplying the vel by this value and it doesn't exist unless there are collisions
            col_margin = cloth.ob.modeling_cloth_self_collision_margin
            sq_margin = col_margin ** 2
            
            # use add.at to get the means of randomly sided polygons    
            cloth.p_means *= 0
            np.add.at(cloth.p_means, cloth.tiler, cloth.co[cloth.p_means_add_tiler])
            cloth.p_means *= cloth.poly_meaner            
            
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
            # normals:
                # could use np.unique to only do normals once then index them with the return index thingy from unique

                # now do closest point edge with points on normals
                nor_vecs = cloth.co[cloth.norm_pairs[p_hits]] - cloth.co[cloth.norm_base[p_hits]][:, nax]
                normals = np.cross(nor_vecs[:,0], nor_vecs[:,1])
                
                base_vecs = cloth.co[v_hits] - cloth.p_means[p_hits]
                d = np.einsum('ij,ij->i', base_vecs, normals) / np.einsum('ij,ij->i', normals, normals)        
                cp = normals * d[:, nax]
                
                # now measure the distance along the normal to see if it's in the cylinder
                
                cp_dot = np.einsum('ij,ij->i', cp, cp)
                in_margin = cp_dot < sq_margin
                
                if in_margin.shape[0] > 0:
                    V3 = v_hits[in_margin]
                    P3 = p_hits[in_margin]
                    cp3 = cp[in_margin]
                    cpd3 = cp_dot[in_margin]
                    
                    d1 = sq_margin
                    d2 = cpd3
                    div = d1/d2
                    surface = cp3 * np.sqrt(div)[:, nax]

                    force = np.nan_to_num(surface - cp3)
                    force *= cloth.ob.modeling_cloth_self_collision_force
                    
                    cloth.co[V3] += force
                    #np.add.at(cloth.co, V3, force * .5)
                    #np.multiply.at(cloth.vel, V3, 0.2)
                    
                    # could get some speed help by iterating over a dict maybe
                    #if False:    
                    if True:    
                        for i in range(len(P3)):

                            #print(i, cloth.v_per_p[i].shape, force.shape)
                            #print(cloth.v_per_p[i], force)
                            cloth.co[cloth.v_per_p[P3[i]]] -= force[i]
                            
                            cloth.vel[cloth.v_per_p[P3[i]]] *= 0.2
                            #cloth.vel[cloth.v_per_p[P3[i]]] += cloth.vel[V3[i]]
                            
                            #need to transfer the velocity back and forth between hit faces.

        #collision=====================================
        

        # calc velocity
        #vel_dif = cloth.vel_start - cloth.co

        #cloth.vel += vel_dif
        
        ##np.multiply.at(cloth.vel, V3, 0)

        
        cloth.vel[:,2][cloth.pindexer] -= cloth.ob.modeling_cloth_gravity * .01
        
        cloth.vel *= cloth.ob.modeling_cloth_velocity
        co[cloth.pindexer] -= cloth.vel[cloth.pindexer]        


        if len(cloth.pin_list) > 0:
            cloth.co[cloth.pin_list] = hook_co
            cloth.vel[cloth.pin_list] = 0
        
        if cloth.clicked: # for the grab tool
            for v in range(len(extra_data['vidx'])):   
                vert = cloth.ob.data.shape_keys.key_blocks['cloth key'].data
                loc = extra_data['stored_vidx'][v] + extra_data['move']
                cloth.co[extra_data['vidx'][v]] = loc            
                cloth.vel[extra_data['vidx'][v]] *= 0

        cloth.ob.data.shape_keys.key_blocks['cloth key'].data.foreach_set('co', cloth.co.ravel())
        cloth.ob.data.shape_keys.key_blocks['cloth key'].mute = True
        cloth.ob.data.shape_keys.key_blocks['cloth key'].mute = False


def modeling_cloth_handler(scene):
    items = data.items()
    if len(items) == 0:
        for i in bpy.data.objects:
            i.modeling_cloth = False
        
        for i in bpy.app.handlers.scene_update_post:
            if i.__name__ == 'modeling_cloth_handler':
                bpy.app.handlers.scene_update_post.remove(i)
    
    for i, cloth in items:    
        run_handler(cloth)


def pause_update(self, context):
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
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    coord = event.mouse_region_x, event.mouse_region_y
    guide = create_giude()

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
            verts = np.array([matrix * obj.data.shape_keys.key_blocks['cloth key'].data[v].co for v in obj.data.polygons[face_index].vertices])
            vecs = verts - np.array(hit_world)
            closest = vidx[np.argmin(np.einsum('ij,ij->i', vecs, vecs))]
            length_squared = (hit_world - ray_origin).length_squared
            if best_obj is None or length_squared < best_length_squared:
                best_length_squared = length_squared
                best_obj = obj
                guide.location = matrix * obj.data.shape_keys.key_blocks['cloth key'].data[closest].co
                extra_data['latest_hit'] = matrix * obj.data.shape_keys.key_blocks['cloth key'].data[closest].co
                extra_data['name'] = obj.name
                extra_data['obj'] = obj
                extra_data['closest'] = closest
                
                if extra_data['just_clicked']:
                    extra_data['just_clicked'] = False
                    best_length_squared = length_squared
                    best_obj = obj
                    name = obj.name
                   

class ModelingClothPin(bpy.types.Operator):
    """Modal ray cast for placing pins"""
    bl_idname = "view3d.modeling_cloth_pin"
    bl_label = "Modeling Cloth Pin"
    def __init__(self):
        bpy.ops.object.select_all(action='DESELECT')    
        extra_data['just_clicked'] = False
        
    def modal(self, context, event):
        bpy.context.window.cursor_set("CROSSHAIR")
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
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
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    coord = event.mouse_region_x, event.mouse_region_y

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
                verts = np.array([matrix * obj.data.shape_keys.key_blocks['cloth key'].data[v].co for v in obj.data.polygons[face_index].vertices])                
                vert = obj.data.shape_keys.key_blocks['cloth key'].data
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
# drag===================================End



class DeletePins(bpy.types.Operator):
    """Delete modeling cloth pins and clear pin list for current object"""
    bl_idname = "object.delete_modeling_cloth_pins"
    bl_label = "Delete Modeling Cloth Pins"
    
    def execute(self, context):
        ob = get_last_object() # returns tuple with list and last cloth objects or None
        if ob is not None:
            data[ob[1].name].pin_list = []
            for i in data[ob[1].name].hook_list:
                bpy.data.objects.remove(i)

            data[ob[1].name].hook_list = []
        bpy.context.scene.objects.active = ob[1]
        return {'FINISHED'}


class SelectPins(bpy.types.Operator):
    """Select modeling cloth pins for current object"""
    bl_idname = "object.select_modeling_cloth_pins"
    bl_label = "Select Modeling Cloth Pins"
    
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
    
    def execute(self, context):
        ob = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        sel = [i.index for i in ob.data.vertices if i.select]
                
        name = ob.name
        matrix = ob.matrix_world.copy()
        matrix_inv = ob.matrix_world.inverted().copy()
        
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
        
    def execute(self, context):
        update_pin_group()
        return {'FINISHED'}


class GrowSource(bpy.types.Operator):
    """Grow Source Shape"""
    bl_idname = "object.modeling_cloth_grow"
    bl_label = "Modeling Cloth Grow"
        
    def execute(self, context):
        scale_source(1.02)
        return {'FINISHED'}


class ShrinkSource(bpy.types.Operator):
    """Shrink Source Shape"""
    bl_idname = "object.modeling_cloth_shrink"
    bl_label = "Modeling Cloth Shrink"
        
    def execute(self, context):
        scale_source(0.98)
        return {'FINISHED'}


class ResetShapes(bpy.types.Operator):
    """Reset Shapes"""
    bl_idname = "object.modeling_cloth_reset"
    bl_label = "Modeling Cloth Reset"
        
    def execute(self, context):
        reset_shapes()
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
        default=False)#, update=pause_update)

    bpy.types.Object.modeling_cloth_self_collision_force = bpy.props.FloatProperty(name="recovery force", 
        description="Self colide faces repel", 
        default=.02, precision=4, min= -1.1, max=1.1, soft_min= 0, soft_max=1)

    bpy.types.Object.modeling_cloth_self_collision_margin = bpy.props.FloatProperty(name="Margin", 
        description="Self colide faces margin", 
        default=.08, precision=4, min= -1, max=1, soft_min= 0, soft_max=1)

    bpy.types.Object.modeling_cloth_self_collision_cy_size = bpy.props.FloatProperty(name="Cylinder size", 
        description="Self colide faces cylinder size", 
        default=1, precision=4, min= 0, max=4, soft_min= 0, soft_max=1.5)


    bpy.types.Scene.modeling_cloth_data_set = {} 
    bpy.types.Scene.modeling_cloth_data_set_extra = {} 

        
def remove_properties():            
    '''Drives to the grocery store and buys a sandwich'''
    del(bpy.types.Object.modeling_cloth)
    del(bpy.types.Object.modeling_cloth_pause)
    del(bpy.types.Object.modeling_cloth_noise)    
    del(bpy.types.Object.modeling_cloth_noise_decay)
    del(bpy.types.Object.modeling_cloth_spring_force)
    del(bpy.types.Object.modeling_cloth_gravity)        
    del(bpy.types.Object.modeling_cloth_iterations)
    del(bpy.types.Object.modeling_cloth_velocity)

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
                col.prop(ob ,"modeling_cloth_velocity", text="Velocity")#, icon='PLAY')        
                col.prop(ob ,"modeling_cloth_floor", text="Floor")#, icon='PLAY')        
                col = layout.column(align=True)
                col.scale_y = 1.5
                col.alert = status
                if ob.modeling_cloth:    
                    if ob.mode == 'EDIT':
                        col.operator("object.modeling_cloth_pin_selected", text="Pin Selected")
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
