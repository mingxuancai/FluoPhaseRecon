import numpy

def coordinate_transform(fx_source, fy_source, shape_size):
    middle = shape_size//2
    fx_source *= middle
    fy_source *= middle
    print(fx_source)
    if fx_source >= 0:
        if fy_source >= 0: # region 4
            fx_source -= middle
            fy_source -= middle
        else: # region 3
            fx_source -= middle
            fy_source += middle
    else:
        if fy_source >= middle: # region 2
            fx_source += middle
            fy_source -= middle
        else: # region 1
            fx_source += middle
            fy_source += middle
    # print(fx_source/shape_size)
    # print(fy_source/shape_size)
    return fx_source/shape_size, fy_source/shape_size