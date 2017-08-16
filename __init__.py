#----------------------------------------------------------
# File __init__.py
#----------------------------------------------------------
 
#    Addon info
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
if "bpy" in locals():
    import imp
    imp.reload(ModelingCloth)
    imp.reload(SurfaceFollow)
    imp.reload(UVShape)
    print("Reloaded multifiles")
else:
    from . import ModelingCloth, SurfaceFollow, UVShape
    print("Imported multifiles")
    



def register():
    ModelingCloth.register()
    SurfaceFollow.register()
    UVShape.register()
 

def unregister():
    ModelingCloth.unregister()
    SurfaceFollow.unregister()
    UVShape.unregister()
     

if __name__ == "__main__":
    register()
