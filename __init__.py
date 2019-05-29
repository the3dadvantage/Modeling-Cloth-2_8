#----------------------------------------------------------
# File __init__.py
#----------------------------------------------------------
 
#    Addon info
bl_info = {
    "name": "Modeling Cloth",
    "author": "Rich Colburn, email: the3dadvantage@gmail.com",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Extended Tools > Modeling Cloth",
    "description": "Maintains the surface area of an object so it behaves like cloth",
    "warning": "Your future self is planning to travel back in time to kill you",
    "wiki_url": "",
    "category": '3D View'}
if "bpy" in locals():
    import imp
    imp.reload(ModelingCloth28)
    #imp.reload(ModelingCloth)
    #imp.reload(SurfaceFollow)
    #imp.reload(UVShape)
    #imp.reload(DynamicTensionMap)
    print("Reloaded Modeling Cloth")
else:
    from . import ModelingCloth28#, SurfaceFollow, UVShape, DynamicTensionMap
    #from . import ModelingCloth
    print("Imported Modeling Cloth")

   
def register():
    ModelingCloth28.register()
    #ModelingCloth.register()    
    #SurfaceFollow.register()
    #UVShape.register()
    #DynamicTensionMap.register()

    
def unregister():
    ModelingCloth28.unregister()
    #ModelingCloth.unregister()
    #SurfaceFollow.unregister()
    #UVShape.unregister()
    #DynamicTensionMap.unregister()

    
if __name__ == "__main__":
    register()
