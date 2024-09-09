# main for the initator
# 16 slices this case
# 1 simple GEMM + Softmax
# batch_size= 4,seq_len= 2048, hidden_size=4096, head_dim=128, head_num=int(hidden_size/head_dim)=32
# total parrale=batch*head_num=4*32=128

# Q=[4,32,2048,128];K=[4,32,128,2048];  ; V=[4,32,2048,128], S*V=[4,32,2048,128]
# O=Q*K=[4,32,2048,2048];
# S=Softmac(O)=[4,32,2048,2048]
# P=S*V=[4,32,2048,128]
# C=Transform(P)=[4,2048,4096]

# for the naive data flow, let's do it now!
# tileQ=[64,128], tileK=[128,128]
# let's generate it now

# generation here

import sys
from hardware_modeling import *


def generate_naive_mha_plan(batch_size, seq_len, hidden_size, head_dim, bM, bN, bK):

    chip = CGRA_VER15()
    dtype = "float16"
    bytes_per_num = 2

    slice_count = chip.sm_count  # 16 actually

    head_num = int(hidden_size/head_dim)

    # head_num=1
    # batch_size=1

    gridM = int(seq_len/bM)
    gridN = int(seq_len/bN)
    gridK = int(head_dim/bK)

    strideM = 4
    strideN = int(slice_count/strideM)

    print(f"---------------------Start of CGRA's Schedule Plan---------------------")

    print(f"make_tensor::<4D>(q,dimension=[{batch_size},{head_num},{seq_len},{head_dim}],"
          f"tile_dimension=[{1},{1},{bM},{bK}],dtype={dtype})")

    print(f"make_tensor::<4D>(k,dimension=[{batch_size},{head_num},{head_dim},{seq_len}],"
          f"tile_dimension=[{1},{1},{bK},{bN}],dtype={dtype})")

    print(f"make_tensor::<4D>(o,dimension=[{batch_size},{head_num},{seq_len},{seq_len}],"
          f"tile_dimension=[{1},{1},{bM},{bN}],dtype={dtype})")

    for index in range(slice_count):
        print(f"slice.{index}.spm_allocate(q_spm0,[{bM},{bK}],{dtype})")
        print(f"slice.{index}.spm_allocate(q_spm1,[{bM},{bK}],{dtype})")
        print(f"slice.{index}.spm_allocate(k_spm0,[{bK},{bN}],{dtype})")
        print(f"slice.{index}.spm_allocate(k_spm1,[{bK},{bN}],{dtype})")
        print(f"slice.{index}.spm_allocate(o_spm0,[{bM},{bN}],{dtype})")
        print(f"slice.{index}.spm_allocate(o_spm1,[{bM},{bN}],{dtype})")

    for index in range(slice_count):
        print(f"slice.{index}.claim_bar(q_bar0,[{bM},{bK}],{dtype})")
        print(f"slice.{index}.claim_bar(q_bar1,[{bM},{bK}],{dtype})")
        print(f"slice.{index}.claim_bar(k_bar0,[{bK},{bN}],{dtype})")
        print(f"slice.{index}.claim_bar(k_bar1,[{bK},{bN}],{dtype})")
        print(f"slice.{index}.claim_bar(o_bar0,[{bM},{bN}],{dtype})")
        print(f"slice.{index}.claim_bar(o_bar1,[{bM},{bN}],{dtype})")

    coords = extract_blocks(gridM, gridN, strideM, strideN)

    # for coord in coords:
    #     print(coord)

    tile_iter = 0
    k_iter = 0
    max_count = int(batch_size*head_num*gridM*gridN*gridK/slice_count)

    store_flag = False

    for batch in range(batch_size):
        # print(batch)
        for head in range(head_num):

            for count in range(0, int(gridM * gridN), slice_count):

                length = min(slice_count, int(gridM * gridN) - count)

                Q_Unique_per_It = [[] for _ in range(gridM)]
                K_Unique_per_It = [[] for _ in range(gridN)]

                slice_tile = []

                for i in range(length):
                    current_idx = count+i
                    m, n = coords[current_idx]-1
                    # print(m,gridM)
                    # print(n,gridN)
                    Q_Unique_per_It[m].append(i)
                    K_Unique_per_It[n].append(i)

                    slice_tile.append([m, n])

                # now in a iteration, tma load mask can be easily infered
                for k in range(gridK):

                    # print("to_store_slice_tile",to_store_slice_tile)

                    # first iteration
                    if (tile_iter) == 0:

                        for m in range(gridM):
                            if not Q_Unique_per_It[m]:
                                pass
                            else:
                                print(f"tma.load.multicast(q[{batch},{head},{m},{k}],q_spm{
                                      tile_iter % 2},mask={Q_Unique_per_It[m]},set_bar=[q_bar{tile_iter % 2}])")

                        for n in range(gridN):
                            if not K_Unique_per_It[n]:
                                pass
                            else:
                                print(f"tma.load.multicast(k[{batch},{head},{k},{n}],k_spm{
                                      tile_iter % 2},mask={K_Unique_per_It[n]},set_bar=[k_bar{tile_iter % 2}])")
                    # 1~max-2 iter
                    else:

                        if (k == 0):
                            store_flag = True
                            to_store_batch = tmp_batch
                            to_store_head = tmp_head
                            to_store_slice_tile = tmp_slice_tile
                            # print("to_store_slice_tile",to_store_slice_tile)

                        for m in range(gridM):
                            if not Q_Unique_per_It[m]:
                                pass
                            else:
                                print(f"tma.load.multicast(q[{batch},{head},{m},{k}],q_spm{
                                      tile_iter % 2},mask={Q_Unique_per_It[m]},set_bar=[q_bar{tile_iter % 2}])")

                        for n in range(gridN):
                            if not K_Unique_per_It[n]:
                                pass
                            else:
                                print(f"tma.load.multicast(k[{batch},{head},{k},{n}],k_spm{
                                      tile_iter % 2},mask={K_Unique_per_It[n]},set_bar=[k_bar{tile_iter % 2}])")

                        for slice_id in range(slice_count):
                            print(f"slice.{slice_id}.gemm::<dtype={dtype},bM={bM},bN={bN},bK={bK},Layout=RowMajor>(q_spm{(tile_iter-1) % 2},k_spm{(tile_iter-1) % 2},o_spm{(k_iter-1) % 2},"
                                  f"wait_bar=[q_bar{(tile_iter-1) % 2},k_bar{(tile_iter-1) % 2}],set_bar=[o_bar{(k_iter-1) % 2}])")

                        if (store_flag == True):
                            store_flag = False
                            for slice_id in range(slice_count):
                                print(f"tma.store.slice.{slice_id}(o_spm{(k_iter-1) % 2},o[{to_store_batch},{to_store_head},{
                                      to_store_slice_tile[slice_id][0]},{to_store_slice_tile[slice_id][1]}],wait_bar=[o_bar{(k_iter-1) % 2}])")

                        # if(k==gridK-1):
                        #     store_flag=True
                        #     to_store_batch=batch
                        #     to_store_head=head
                        #     to_store_slice_tile=slice_tile
                        #     print("to_store_slice_tile",to_store_slice_tile)

                    # last iter
                        if (tile_iter == max_count-1):
                            for slice_id in range(slice_count):
                                print(f"slice.{slice_id}.gemm::<dtype={dtype},bM={bM},bN={bN},bK={bK},Layout=RowMajor>(q_spm{(tile_iter) % 2},k_spm{(tile_iter) % 2},o_spm0,"
                                      f"wait_bar=[q_bar{(tile_iter) % 2},k_bar{(tile_iter) % 2}],set_bar=[o_bar{k_iter % 2}])")

                            # if (k==0):
                            #     store_flag=True
                            #     to_store_batch=tmp_batch
                            #     to_store_head=tmp_head
                            #     to_store_slice_tile=tmp_slice_tile
                            #     # print("to_store_slice_tile",to_store_slice_tile)
                            store_flag = True

                            if (store_flag == True):
                                store_flag = True
                                for slice_id in range(slice_count):
                                    print(f"tma.store.slice.{slice_id}(o_spm{(k_iter-0) % 2},o[{to_store_batch},{to_store_head},{
                                          to_store_slice_tile[slice_id][0]},{to_store_slice_tile[slice_id][1]}],wait_bar=[o_bar{(k_iter-0) % 2}])")

                            # if(store_flag==True):
                            #     store_flag=False
                            #     for slice_id in range(slice_count):
                            #         print(f"tma.store.slice{slice_id}.(o_spm0,o[{batch},{head},{slice_tile[slice_id][0]},{slice_tile[slice_id][1]}],wait_bar=[o_bar{(k_iter)%2}])")

                    if (k == gridK-1):
                        tmp_batch = batch
                        tmp_head = head
                        tmp_slice_tile = slice_tile

                    tile_iter += 1

                k_iter += 1

                # for slice_id in range(slice_count):
                #     print(f"tma.store.slice{slice_id}.(o_spm0,o[{batch},{head},{slice_tile[slice_id][0]},{slice_tile[slice_id][1]}],wait_bar=[o_bar{(tile_iter-1)%2}])")


# -----------------------
    # for count in range(0, int(gridM * gridN), slice_count):

    #     length = min(slice_count, int(gridM * gridN) - count)

    #     Q_Unique_per_It = [[] for _ in range(gridM)]
    #     K_Unique_per_It = [[] for _ in range(gridN)]

    #     for i in range(length):
    #         current_idx=count+i
    #         m, n = coords[current_idx]-1
    #         # print(m,gridM)
    #         # print(n,gridN)
    #         Q_Unique_per_It[m].append(i)
    #         K_Unique_per_It[n].append(i)

    #     # now in a iteration, tma load mask can be easily infered

    #     for m in range(gridM):
    #         if not Q_Unique_per_It[m]:
    #             pass
    #         else:

    #             #
    #             print(f"tma.load.multicast(q[{batch},{head},{m},{0}],q_spm{0},mask={Q_Unique_per_It[m]})")
    #             print(f"tma.load.multicast(k[{batch},{head},{0},{n}],k_spm{0},mask={K_Unique_per_It[n]})")
    #             pass
# -----------------------

    for index in range(slice_count):
        print(f"slice.{index}.spm_free(q_spm0)")
        print(f"slice.{index}.spm_free(q_spm1)")
        print(f"slice.{index}.spm_free(k_spm0)")
        print(f"slice.{index}.spm_free(k_spm1)")
        print(f"slice.{index}.spm_free(o_spm0)")
        print(f"slice.{index}.spm_free(o_spm1)")

    print(f"---------------------End of CGRA's Schedule Plan---------------------")

    # pass
if __name__ == '__main__':
    batch_size = 1
    seq_len = 1024
    hidden_size = 1024
    head_dim = 128
    bM = 64
    bN = 128
    bK = 128

    sys.stdout = open("resource/program.txt", "w")

    generate_naive_mha_plan(batch_size, seq_len,
                            hidden_size, head_dim, bM, bN, bK)
