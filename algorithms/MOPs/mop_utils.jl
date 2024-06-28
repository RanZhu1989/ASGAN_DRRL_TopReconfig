# utils for building environment

function read_data(sys_file_name, scenario_file_name)

    
    Sb = 100                   # MW/MVar/MVA
    Vb = 230               # kV
    Zb = Vb^2/Sb              # O
    # Ib = Sb/(sqrt(3)*Vb)      # kA
    V0 = 1                    # p.u.
    V_max = 1.05
    V_min = 0.95
    Big_M_FF = 200            # BigM of the single commodity flow
    Big_M_V = 3               # BigM of the voltage
    BigM_SC = 2

    NT=24
    # Read data from excel
    Bus_Data = XLSX.readtable(joinpath(@__DIR__,sys_file_name), "Bus")
    Bus_Data = hcat(Bus_Data.data ...)

    DG_Data = XLSX.readtable(joinpath(@__DIR__,sys_file_name), "DG")
    DG_Data = hcat(DG_Data.data ...)

    Branch_Data = XLSX.readtable(joinpath(@__DIR__,sys_file_name), "Branch")
    Branch_Data = hcat(Branch_Data.data ...)

    N_Bus = 124
    Pd_Data = Bus_Data[:, 2] ./ Sb
    Pd_Data_all = sum(Pd_Data)
    Qd_Data = Bus_Data[:, 3] ./ Sb
    N_NoLoad_Bus = sum(Pd_Data .== 0)
    Ind_NoLoad_Bus = findall(x -> x == 0, Pd_Data)
    Mask_NoLoad_Bus = MakeMask(N_Bus, N_NoLoad_Bus, Ind_NoLoad_Bus)
    Qd_Data_all = sum(Qd_Data)
    Pd_ratio = Pd_Data ./ Pd_Data_all    # Active load ratio
    Pd_ratio = reshape(Pd_ratio, length(Pd_ratio), 1, 1)
    Qd_ratio = Qd_Data ./ Qd_Data_all    # Reactive load ratio


    N_Branch = size(Branch_Data, 1)   
    N_TL = sum(Branch_Data[:, 7] .== 1)
    Ind_TL = findall(x -> x == 1, Branch_Data[:, 7])   # index of tie-line
    TL_Mask = MakeMask(N_Branch, N_TL, Ind_TL)   # mask of tie-line
    N_NL = sum(Branch_Data[:, 7] .== 0)
    Ind_NL = findall(x -> x == 0, Branch_Data[:, 7])   # index of non-tie-line
    NL_Mask = MakeMask(N_Branch, N_NL, Ind_NL)   # mask of non-tie-line
    Branch_start = Branch_Data[:, 2]   # start and end node of each branch
    Branch_end = Branch_Data[:, 3]
    Vulne_branch_all = Branch_Data[:, 8]   # vulnerability of each branch
    Ind_vulne_branch = findall(x -> x == 1, Vulne_branch_all)   # index of vulnerable branch
    N_vulne_branch = size(Ind_vulne_branch, 1)   # number of vulnerable branch
    Vulne_branch_mask = MakeMask(N_Branch,N_vulne_branch, Ind_vulne_branch) # mask of vulnerable branch

    # LS_Mask = MakeMask(N_Bus, N_Branch, Branch_start)
    # LE_Mask = MakeMask(N_Bus, N_Branch, Branch_end)
    pIn = MakeIncMatrix(Branch_start, Branch_end)   # node-branch incidence matrix  jk-ij
    pInn = copy(pIn)
    pInn[pInn .> 0] .= 0    # Inn is the negative part of I   +ij
    R_Branch0 = Branch_Data[:, 4] ./ Zb   # Resistance of each branch
    X_Branch0 = Branch_Data[:, 5] ./ Zb   # Reactance of each branch
    SZ_Branch0 = R_Branch0 .^ 2 + X_Branch0 .^ 2   # Square of impedance of each branch
    S_Branch0 = Branch_Data[:, 6] ./ Sb

    N_DG = size(DG_Data, 1)
    DataDN_IndDG = DG_Data[:, 2]   # DG Location
    DataDN_IndBSDG = DG_Data[findall(x -> x == 1, DG_Data[:, 7]), 2]   # Black start DG
    # DataDN_IndNMDG = DG_Data[findall(x -> x == 0, DG_Data[:, 7]), 2]   # Non-black start DG
    N_BSDG = size(DataDN_IndBSDG, 1)
    # N_NMDG = size(DataDN_IndNMDG, 1)
    P_DG_max0 = DG_Data[:, 3] ./ Sb
    P_DG_min0 = DG_Data[:, 4] ./ Sb
    Q_DG_max0 = DG_Data[:, 5] ./ Sb
    Q_DG_min0 = DG_Data[:, 6] ./ Sb
    DG_Mask = MakeMask(N_Bus, N_DG, DataDN_IndDG)   # Mask matrix of DG Location
    BSDG_Mask = MakeMask(N_Bus, N_BSDG, DataDN_IndBSDG)   # Mask matrix of BSDG Location
    # NMDG_Mask = MakeMask(N_Bus, N_NMDG, DataDN_IndNMDG)  
    # Read time-series data
    Load_Scenario_Data = XLSX.readtable(joinpath(@__DIR__,scenario_file_name), "load",header=false)
    Load_Scenario_Data = hcat(Load_Scenario_Data.data ...)

    DG_Scenario_Data = XLSX.readtable(joinpath(@__DIR__,scenario_file_name), "DG",header=false )
    DG_Scenario_Data = hcat(DG_Scenario_Data.data ...)
    DG_ratio = reshape(DG_Scenario_Data, size(DG_Scenario_Data,1) , NT, N_DG)
    DG_ratio = permutedims(DG_ratio, [3, 2, 1]) # 12,24,300

    Disruption_Scenario_Data = XLSX.readtable(joinpath(@__DIR__,scenario_file_name), "disruption",header=false)
    Disruption_Scenario_Data = hcat(Disruption_Scenario_Data.data ...)
    Disruption_factor = reshape(Disruption_Scenario_Data, size(Disruption_Scenario_Data,1) , N_vulne_branch, NT) # 300,5,24
    Disruption_factor = permutedims(Disruption_factor, [2, 3, 1])


    NS = size(Load_Scenario_Data, 1)   # scenario number

    P_DG_max = repeat(P_DG_max0, outer=[1, NT, NS]) .* DG_ratio
    Q_DG_max = repeat(Q_DG_max0, outer=[1, NT, NS]) .* DG_ratio 

    load_pec = Load_Scenario_Data # 300,24
    Pd_all = Pd_Data_all .* load_pec  # ratio of total load
    Pd_all = reshape(Pd_all, 1, size(Pd_all, 1), size(Pd_all, 2))
    Qd_all = Qd_Data_all .* load_pec
    Qd_all = reshape(Qd_all, 1, size(Qd_all, 1), size(Qd_all, 2))
    Pd = repeat(Pd_all,outer=[N_Bus,1,1]) .* repeat(Pd_ratio, outer=[1, NS, NT])
    Pd = permutedims(Pd, [1, 3, 2])  # 124,24,300
    Qd = repeat(Qd_all,outer=[N_Bus,1,1]) .* repeat(Qd_ratio, outer=[1, NS, NT]) # 124,300,24
    Qd = permutedims(Qd, [1, 3, 2])

    
    atk = mapslices(x -> Vulne_branch_mask * x, Disruption_factor, dims=1)  # 5,24,300
    atk = 1 .- atk

    args = (NT, NS, N_TL, N_NL, N_vulne_branch, Branch_start, N_Branch, TL_Mask, NL_Mask, N_Bus, pIn, pInn, N_DG, DG_Mask, R_Branch0, X_Branch0, Big_M_V, V0,
            V_min, V_max, Pd, Qd, SZ_Branch0, S_Branch0, P_DG_min0, P_DG_max, Q_DG_min0, Q_DG_max, BigM_SC, BSDG_Mask, Mask_NoLoad_Bus,
            Big_M_FF, atk, Sb)

    return args
    
end

# pdn data
function MakeMask(x, y, list)
    Mask = zeros(x, y)
    for i = 1:x
        Mask[i, findall(x -> x == i, list)] .= 1
    end
    return Mask
end

function MakeIncMatrix(s, t)
    MaxNode = max(maximum(s), maximum(t))
    Inc = zeros(MaxNode, length(s))
    for j = 1:length(s)
        Inc[s[j], j] = 1
        Inc[t[j], j] = -1
    end
    return Inc
end