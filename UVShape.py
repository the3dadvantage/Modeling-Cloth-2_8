import bpy
import numpy as np

bl_info = {
    "name": "UV Shape",
    "author": "Rich Colburn, email: the3dadvantage@gmail.com",
    "version": (1, 0),
    "blender": (2, 78, 0),
    "location": "View3D > Extended Tools > Create UV Shape",
    "description": "Creates a flattened version of the mesh as a shape key",
    "warning": "Experimental: Everyone who has ever died did so while alive",
    "wiki_url": "",
    "category": '3D View'}


#################################
def get_selected_edges(ob='empty'):
    '''returns a bool array of selected edges'''
    if ob == 'empty':
        ob = bpy.context.object
    ed = np.zeros(len(ob.data.edges), dtype=np.bool)
    ob.data.edges.foreach_get('select', ed)
    return ed

def get_edge_idx(ob='empty'):
    if ob == 'empty':
        ob = bpy.context.object
    ed = np.zeros(len(ob.data.edges)*2, dtype=np.int32)
    ob.data.edges.foreach_get('vertices', ed)
    return ed.reshape(len(ed)//2, 2)

def get_coords(ob='empty', proxy=False):
    '''Creates an N x 3 numpy array of vertex coords. If proxy is used the
    coords are taken from the object specified with modifiers evaluated.
    For the proxy argument put in the object: get_coords(ob, proxy_ob)'''
    if ob == 'empty':
        ob = bpy.context.object
    if proxy:
        mesh = proxy.to_mesh(bpy.context.scene, True, 'PREVIEW')
        verts = mesh.vertices
    else:
        verts = ob.data.vertices
    v_count = len(verts)
    coords = np.zeros(v_count * 3)
    verts.foreach_get("co", coords)
    if proxy:
        bpy.data.meshes.remove(mesh)
    return coords.reshape(v_count, 3)

def get_key_coords(ob='empty', key='Basis', proxy=False):
    '''Creates an N x 3 numpy array of vertex coords.from
    shape keys'''
    if ob == 'empty':
        ob = bpy.context.object
    if proxy:
        mesh = proxy.to_mesh(bpy.context.scene, True, 'PREVIEW')
        verts = mesh.data.shape_keys.key_blocks[key].data
    else:
        verts = ob.data.shape_keys.key_blocks[key].data
    v_count = len(verts)
    coords = np.zeros(v_count * 3)
    verts.foreach_get("co", coords)
    if proxy:
        bpy.data.meshes.remove(mesh)
    return coords.reshape(v_count, 3)

def set_coords(coords, ob='empty', use_proxy='empty'):
    """Writes a flattened array to the object. Second argument is for reseting
    offsets created by modifiers to avoid compounding of modifier effects"""
    if ob == 'empty':
        ob = bpy.context.object
    if use_proxy == 'empty':    
        ob.data.vertices.foreach_set("co", coords.ravel())
    else:        
        coords += use_proxy        
        ob.data.vertices.foreach_set("co", coords.ravel())
    ob.data.update()

def get_uv_coords(ob='empty', layer='UV_Shape_key', proxy=False):
    '''Creates an N x 2 numpy array of uv coords. If proxy is used the
    coords are taken from the object specified with modifiers evaluated.
    For the proxy argument put in the object: get_coords(ob, proxy_ob)
    "layer='my_uv_map_name'" is the name of the uv layer you want to use.'''
    if ob == 'empty':
        ob = bpy.context.object
    if proxy:
        mesh = proxy.to_mesh(bpy.context.scene, True, 'PREVIEW')
        verts = mesh.uv_layers[layer].data
    else:
        verts = ob.data.uv_layers[layer].data
    v_count = len(verts)
    coords = np.zeros(v_count * 2)
    verts.foreach_get("uv", coords)
    if proxy:
        bpy.data.meshes.remove(mesh)
    return coords.reshape(v_count, 2)

def total_length(ed='empty', coords='empty', ob='empty'):
    '''Returns the total length of all edge segments'''
    if ob == 'empty':
        ob = bpy.context.object
    if coords == 'empty':    
        coords = get_coords(ob)
    if ed == 'empty':    
        ed = get_edge_idx(ob)
    edc = coords[ed]
    e1 = edc[:, 0]
    e2 = edc[:, 1]
    ee = e1 - e2
    leng = np.einsum('ij,ij->i', ee, ee)
    return np.sum(np.sqrt(leng))

def total_length_selected(ed='empty', coords='empty', ob='empty'):
    '''Returns the total length of all edge segments'''
    if ob == 'empty':
        ob = bpy.context.object
    if coords == 'empty':    
        coords = get_coords(ob)
    if ed == 'empty':    
        ed = get_edge_idx(ob)
    edc = coords[ed]
    e1 = edc[:, 0]
    e2 = edc[:, 1]
    ee1 = e1 - e2
    sel = get_selected_edges(ob)    
    ee = ee1[sel]    
    leng = np.einsum('ij,ij->i', ee, ee)
    return np.sum(np.sqrt(leng))

def basic_unwrap():
    ob = bpy.context.object
    mode = ob.mode
    data = ob.data
    key = ob.active_shape_key_index
    bpy.ops.object.mode_set(mode='OBJECT')        
    layers = [i.name for i in ob.data.uv_layers]
    if "UV_Shape_key" not in layers:
        bpy.ops.mesh.uv_texture_add()
        ob.data.uv_layers[len(ob.data.uv_layers) - 1].name = 'UV_Shape_key'
    
    ob.data.uv_layers.active_index = len(ob.data.uv_layers) - 1
    ob.active_shape_key_index = 0
    data.vertices.foreach_set('select', np.ones(len(data.vertices), dtype=np.bool))

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.0635838)
    bpy.ops.object.mode_set(mode=mode)
    ob.active_shape_key_index = key



def get_piece_bool(num, dict):
    '''Uses a vertex number to find the right bool array
    as created by divide_garment()'''
    count = 0
    nums = dict['garment_pieces']['numbers_array']    
    for i in nums:
        if np.in1d(num, i):
            return count
        count += 1        

def find_linked(ob, vert, per_face='empty'):
    '''Takes a vert and returns an array of linked face indices'''
    the_coffee_is_hot = True
    fidx = np.arange(len(ob.data.polygons))
    eidx = np.arange(len(ob.data.edges))
    f_set = np.array([])
    e_set = np.array([])
    verts = ob.data.vertices
    verts[vert].select = True
    v_p_f_count = [len(p.vertices) for p in ob.data.polygons]
    max_count = np.max(v_p_f_count)
    if per_face == 'empty':    
        per_face = [[i for i in poly.vertices] for poly in ob.data.polygons]
    for i in per_face:
        for j in range(max_count-len(i)):
            i.append(i[0])
    verts_per_face = np.array(per_face)
    vert=np.array([vert])
    
    while the_coffee_is_hot:
        booly = np.any(np.in1d(verts_per_face, vert).reshape(verts_per_face.shape), axis=1)
        f_set = np.append(f_set, fidx[booly])
        new_verts = verts_per_face[booly].ravel()
        if len(new_verts) == 0:
            return np.array(f_set, dtype=np.int64)
            
        cull = np.in1d(new_verts, vert)
        vert = new_verts[-cull]
        verts_per_face = verts_per_face[-booly]
        fidx = fidx[-booly]
    
def divide_garment(ob, dict):
    '''Creates a set of bool arrays and a set of number arrays
    for indexing a sub set of the uv coords. The nuber arrays can
    be used to look up wich bool array to use based on a vertex number'''
    if ob == 'empty':
        ob = bpy.context.object    
    #-----------------------------------    
    v_count = len(ob.data.vertices)
    idx = np.arange(v_count)
    full_set = np.array([])
    dict['islands'] = []
    v_list = [[i for i in poly.vertices] for poly in ob.data.polygons]
    v_in_faces = np.hstack(v_list)
    dict['v_in_faces'] = v_in_faces
    remaining = [1]
    vert = 0
    while len(remaining) > 0:
        linked = find_linked(ob, vert, v_list)
        selected = np.unique(np.hstack(np.array(v_list)[linked]).ravel())
        dict['islands'].append(selected)
        full_set = np.append(full_set, selected)
        remain_bool = np.in1d(idx, full_set, invert=True)
        remaining = idx[remain_bool] 
        if len(remaining) == 0:
            break
        vert = remaining[0]
#################################

def uv_to_shape_key(ob='empty', uv_layer='UV_Shape_key', adjust=True):
    '''Takes the active uv layer and creates a shape key
    uv_layer is a string that can be the name of a uv layer
    otherwise the active uv layer will be used'''    
    bpy.types.Scene.uv_to_shape_dict = {}    
    dict = bpy.context.scene.uv_to_shape_dict
    if ob == 'empty':
        ob = bpy.context.object
    mode=ob.mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(use_extend=False, type='VERT')
    bpy.ops.mesh.select_mode(use_extend=True, type='EDGE')
    bpy.ops.mesh.select_loose()
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.mesh.select_all(action='SELECT')
    #bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')
    divide_garment(ob, dict)
    if not bpy.context.scene.use_active_uv_for_shape:    
        basic_unwrap()

    uv = ob.data.uv_layers
    bpy.context.scene.update()
    if len(uv) > 0:
        if bpy.context.scene.use_active_uv_for_shape:
            idx = uv.active_index
            uv_layer = uv[idx].name
        if ob.data.shape_keys == None:
            ob.shape_key_add('Basis')    
        if 'UV_Shape_key' not in ob.data.shape_keys.key_blocks:
            ob.shape_key_add('UV_Shape_key')
        uv_co = get_uv_coords(ob, uv_layer, proxy=False)
        uv_list = []
        face_verts = dict['v_in_faces']
        uv_arange = np.arange(len(uv_co))
        for i in range(len(ob.data.vertices)):
            x = uv_co[uv_arange[face_verts == i][0]]
            uv_list.append(x)
        uv_co = np.array(uv_list) 
        ins = np.insert(uv_co,2,0, axis=1)
        x_start = 0
        y_min = 0
        if adjust:
            coords = get_coords(ob)
            edge_idx = get_edge_idx(ob)
            for i in dict['islands']:     
                ed_idx = np.in1d(edge_idx, i)
                island_edges = edge_idx[ed_idx[0::2]]
                base_length = total_length(island_edges, coords, ob)
                shape_length = total_length(island_edges, ins, ob)
                div = base_length / shape_length    
                ins[i] *= div
                dif = coords[i] - ins[i]
                ins[i] += np.mean(dif, axis=0)
                x_all = ins[i][:,0]

                x_max = np.max(x_all)
                x_min = np.min(x_all)
                move = x_start - x_min
                x_start = x_max + move + .4
                ins[i] += np.array([move, 0, 0]) 
                

                x_min = np.min(x_all)
                
        ins[:, 2] = 0.1        
        ob.data.shape_keys.key_blocks['UV_Shape_key'].data.foreach_set('co', ins.ravel()) 
        bpy.ops.object.mode_set(mode=mode)
    else:
        print('there was no uv map found')
        bpy.ops.object.mode_set(mode=mode)        
        return None

def do():
    coords = get_key_coords(bpy.context.object, 'UV_Shape_key')
    base = (total_length(ed='empty', coords='empty', ob='empty'))
    map = (total_length(ed='empty', coords=coords, ob='empty'))
    base_sel = (total_length_selected(ed='empty', coords='empty', ob='empty'))
    map_sel = (total_length_selected(ed='empty', coords=coords, ob='empty'))
    bpy.types.Scene.scale = base/map

def update_line_lengths():
    line_lengths(bpy.context.object)

def line_lengths(ob):
    if ob.type == 'MESH':
        if ob.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.mode_set(mode='EDIT')

        sce = bpy.context.scene
        edge_idx = get_edge_idx(ob)
        coords = get_coords(ob)
        sce.base_select_length = total_length_selected(edge_idx, coords, ob)
        
        if len(ob.data.uv_layers) > 0:    
            k_coords = get_key_coords(ob, 'UV_Shape_key')
            sce.shape_select_length = total_length_selected(edge_idx, k_coords, ob)
            div = sce.base_select_length / sce.shape_select_length    
            dif = sce.base_select_length - sce.shape_select_length    
            sce.shape_base_divisor = div
            sce.shape_base_difference = dif
    else:
        print('No active mesh')

def relative_scale(ob='empty'):
    '''Changes the relative scale of the object'''
    if ob == 'empty':
        ob = bpy.context.object
    coords = get_coords(ob) * ob.scale.x
    scale = ob.scale.x
    set_coords(coords * scale, ob)    
    ob.scale = np.tile(1, 3)
    ob.scale = np.tile(ob.relative_scale, 3)
    set_coords(coords / ob.relative_scale, ob)
    
def update_relative(self, context):
    relative_scale()

def create_uv_shape():
    uv_to_shape_key()

class ShapeFromUV(bpy.types.Operator):
    '''Takes the active uv map and makes a shape key'''
    bl_idname = "object.shape_from_uv"
    bl_label = "shape from uv"
    def execute(self, context):
        create_uv_shape()
        return {'FINISHED'}

class UpdateLineLengths(bpy.types.Operator):
    '''Reads current mesh and updates scene properties'''
    bl_idname = "object.update_line_lengths"
    bl_label = "update_line_lengths"
    def execute(self, context):
        update_line_lengths()
        return {'FINISHED'}

def create_properties():
    bpy.types.Scene.base_select_length = bpy.props.FloatProperty(name="Base Select Length", 
        description="Holds the length from the last time Line Length was used", 
        default=0.0)

    bpy.types.Scene.shape_select_length = bpy.props.FloatProperty(name="Shape Select Length", 
        description="Holds the length from the last time Line Length was used", 
        default=0.0)

    bpy.types.Scene.shape_base_difference = bpy.props.FloatProperty(name="Difference", 
        description="Difference in length from shape to base", 
        default=0.0)

    bpy.types.Scene.shape_base_divisor = bpy.props.FloatProperty(name="Divisor", 
        description="Multiply by this value to make them match", 
        default=0.0)

    bpy.types.Scene.shape_select_diameter = bpy.props.FloatProperty(name="Select Diameter", 
        description="Holds the diameter from the last time Line Length was used", 
        default=0.0)

    bpy.types.Object.relative_scale = bpy.props.FloatProperty(name="Relative Scale", 
        description="Changes the relative scale of an object", 
        default=1.0, precision=7, update=update_relative)

    bpy.types.Scene.use_active_uv_for_shape = bpy.props.BoolProperty(name="Use Active", 
        description="Create shape from active uv map. Otherwise generate new", 
        default=False)

def remove_properties():
    """It's never a good idea to clean your marble collection while skydiving"""
    del(bpy.types.Scene.base_select_length)
    del(bpy.types.Scene.shape_select_length)
    del(bpy.types.Scene.shape_base_difference)
    del(bpy.types.Scene.shape_base_divisor)
    del(bpy.types.Scene.shape_select_diameter) 
    del(bpy.types.Object.relative_scale)
    

class Print3DTools(bpy.types.Panel):
    """Creates a new tab with physics UI"""
    bl_label = "3D Print Tools"
    bl_idname = "3D Print Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Extended Tools"
    
    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.label(text="UV Tools")
        col.operator("object.shape_from_uv", text="Create UV Shape", icon='SHAPEKEY_DATA')
        col.prop(bpy.context.scene, "use_active_uv_for_shape", text="Use Active Map", icon='OUTLINER_OB_LATTICE')
        #col.operator("object.update_line_lengths", text="Update Measurements", icon='FILE_REFRESH')
        #col.prop(bpy.context.scene, "base_select_length", text="Base Select Length", icon='FORCE_HARMONIC')                    
        #col.prop(bpy.context.scene, "shape_select_length", text="Shape Select Length", icon='FORCE_HARMONIC')                    
        #col.prop(bpy.context.scene, "shape_base_difference", text="Difference", icon='FORCE_HARMONIC')
        #col.prop(bpy.context.scene, "shape_base_divisor", text="Divisor", icon='FORCE_HARMONIC')
        #if bpy.context.object != None:
            #col.prop(bpy.context.object, "relative_scale", text="Relative Scale", icon='FORCE_HARMONIC')                    

        
def register():
    create_properties()
    bpy.utils.register_class(Print3DTools)
    bpy.utils.register_class(ShapeFromUV)
    bpy.utils.register_class(UpdateLineLengths)

def unregister():
    remove_properties()
    bpy.utils.unregister_class(Print3DTools)
    bpy.utils.unregister_class(ShapeFromUV)
    bpy.utils.unregister_class(UpdateLineLengths)
    
if __name__ == "__main__":
    register()
