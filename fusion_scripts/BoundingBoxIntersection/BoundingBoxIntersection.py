#Author-Eagon
#Description-Using projection sketches from 2DProjection, creates geometry through intersection

import adsk.core, adsk.fusion, adsk.cam, traceback

import adsk.core, adsk.fusion, adsk.cam, traceback

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        product = app.activeProduct
        design = adsk.fusion.Design.cast(product)
        rootComp = design.rootComponent
        bodies = rootComp.bRepBodies
        body = bodies.item(0)
        sketches = rootComp.sketches
        xy_sketch = sketches.add(rootComp.xYConstructionPlane)
        yz_sketch = sketches.add(rootComp.yZConstructionPlane)
        xz_sketch = sketches.add(rootComp.xZConstructionPlane)

        all_aspects = [xy_sketch, yz_sketch, xz_sketch]

        for face in body.faces:
            for aspect in all_aspects:
                aspect.project(face)
        
        ui.messageBox('BBox:{}, {}'.format(body.boundingBox.minPoint,body.boundingBox.maxPoint))
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
