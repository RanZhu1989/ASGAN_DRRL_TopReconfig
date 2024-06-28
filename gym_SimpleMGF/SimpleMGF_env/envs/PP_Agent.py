import pandapower as pp
import pandas as pd
import numpy as np
import pandapower.plotting as plot
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


class Pandapower_Agent():

    def __init__(self,
                 network_xlsx_name,
                 scenario_xlsx_name,
                 v_min=0.95,
                 v_max=1.05):

        # Voltage limits
        self.v_min = v_min
        self.v_max = v_max
        '''
        1. Create the Pandpower Object from the xlsx file
        '''
        # Create an empty network
        self.network = pp.create_empty_network()

        # Create the network from the xlsx file
        # Bus
        df_bus = pd.read_excel(network_xlsx_name,
                               sheet_name='bus_data',
                               index_col=0)
        self.num_bus = df_bus.shape[0]
        df_geo = pd.read_excel(network_xlsx_name,
                               sheet_name='geo_data',
                               index_col=0)  # Geographical data for buses
        bus_geo_list = []
        for i in df_geo.index:
            bus_geo_list.append((df_geo.at[i, 'x'], df_geo.at[i, 'y']))
        for i in df_bus.index:
            pp.create_bus(self.network,
                          vn_kv=df_bus.at[i, 'vn_kv'],
                          max_vm_pu=df_bus.at[i, 'max_vm_pu'],
                          min_vm_pu=df_bus.at[i, 'min_vm_pu'],
                          geodata=bus_geo_list[i])

        # Slack bus
        df_extgrid = pd.read_excel(network_xlsx_name,
                                   sheet_name='slack_data',
                                   index_col=0)
        for i in df_extgrid.index:
            pp.create_ext_grid(self.network,
                               bus=df_extgrid.at[i, 'bus'],
                               vm_pu=df_extgrid.at[i, 'vm_pu'],
                               va_degree=df_extgrid.at[i, 'va_degree'],
                               max_p_mw=df_extgrid.at[i, 'max_p_mw'],
                               min_p_mw=df_extgrid.at[i, 'min_p_mw'],
                               max_q_mvar=df_extgrid.at[i, 'max_q_mvar'],
                               min_q_mvar=df_extgrid.at[i, 'min_q_mvar'],
                               in_service=df_extgrid.at[i, 'in_service'])

        # Line
        df_line = pd.read_excel(network_xlsx_name,
                                sheet_name='line_data',
                                index_col=0)
        for i in df_line.index:
            # Since the order follows the one in the definition
            pp.create_line_from_parameters(self.network, *df_line.loc[i, :])
        self.num_line = df_line.shape[0]
        # Vulnerable lines
        self.vulne_line_list = df_line[df_line['name'].str.contains(
            'Vulne')].index
        self.num_vulne_line = len(self.vulne_line_list)

        # Switch
        df_switch = pd.read_excel(network_xlsx_name,
                                  sheet_name='switch_data',
                                  index_col=0)
        self.num_switch = df_switch.shape[0]
        for i in df_switch.index:
            pp.create_switch(self.network,
                             bus=df_switch.at[i, 'bus'],
                             element=df_switch.at[i, 'element'],
                             et=df_switch.at[i, 'et'],
                             type=df_switch.at[i, 'type'])

        # Initialize the LAST switch states
        self.update_last_switch_states(
            'push')  # For checking the illegal actions

        # Load
        df_load = pd.read_excel(network_xlsx_name,
                                sheet_name='load_data',
                                index_col=0)
        #NOTE: Assume that all loads are critical
        df_load = df_load[~((df_load['p_mw'] == 0) & (df_load['q_mwar'] == 0))]
        self.load_bus_list = df_load['bus'].tolist()
        self.num_load = df_load.shape[0]
        for i in df_load['bus']:
            pp.create_load(self.network,
                           bus=df_load.at[i, 'bus'],
                           p_mw=df_load.at[i, 'p_mw'],
                           q_mvar=df_load.at[i, 'q_mwar'])

        # Sgen := PV + WP
        df_sgen = pd.read_excel(network_xlsx_name,
                                sheet_name='sgen_data',
                                index_col=0)
        self.num_sgen = df_sgen.shape[0]
        self.sgen_bus_list = df_sgen['bus'].tolist()
        self.pv_list = df_sgen[df_sgen['name'].str.contains('PV')].index
        self.wp_list = df_sgen[df_sgen['name'].str.contains('WP')].index
        self.pv_num = len(self.pv_list)
        self.wp_num = len(self.wp_list)
        for i in df_sgen.index:
            pp.create_sgen(self.network,
                           bus=df_sgen.at[i, 'bus'],
                           p_mw=df_sgen.at[i, 'p_mw'],
                           q_mvar=df_sgen.at[i, 'q_mvar'],
                           name=df_sgen.at[i, 'name'])
        '''
        2. Pre-load the scenario data
        '''
        self.load_scenario = pd.read_excel(scenario_xlsx_name,
                                           sheet_name='load',
                                           header=None).to_numpy()
        # total number of time slots
        self.T = self.load_scenario.shape[1]
        # total number of scenarios
        self.S = self.load_scenario.shape[0]

        # Read scenario data
        self.pv_scenario = pd.read_excel(scenario_xlsx_name,
                                         sheet_name='pv',
                                         header=None).to_numpy()
        # reshape the data to S*T*N, where N is the number of PVs or WPs or vulnerable lines
        # E.g., pv1_t1, pv2_t1, ..., pv1_t2, pv2_t2, ...
        #      |----t1--------| ... |----t2--------| ...
        self.pv_scenario = self.pv_scenario.reshape(self.S, self.T, self.pv_num)
        self.wp_scenario = pd.read_excel(scenario_xlsx_name,
                                         sheet_name='wp',
                                         header=None).to_numpy()
        self.wp_scenario = self.wp_scenario.reshape(self.S, self.T, self.wp_num)
        self.disruption_scenario = pd.read_excel(scenario_xlsx_name,
                                                 sheet_name='disruption',
                                                 header=None).to_numpy()
        # 取反
        self.disruption_scenario = 1 - self.disruption_scenario
        self.disruption_scenario = self.disruption_scenario.reshape(
            self.S, self.T, self.num_vulne_line)

        # Check the shape of the scenario data
        if self.pv_scenario.shape[1]*self.pv_scenario.shape[2] != self.T*self.pv_num \
            or self.wp_scenario.shape[1]*self.wp_scenario.shape[2] != self.T*self.wp_num \
            or self.disruption_scenario.shape[1]*self.disruption_scenario.shape[2] != self.T*self.num_vulne_line:
            raise ValueError(
                'The number of columns in the scenario data is not correct.')

    def ts_assign(self, scenario_id, time_step):
        '''
        Assign the scenario data to the network
        Return scenario_configs
        '''
        # Assign the load
        self.network.load['scaling'] = self.load_scenario[scenario_id,
                                                          time_step]

        # Assign the PV
        for i, idx in enumerate(self.pv_list):
            self.network.sgen.at[idx,
                                 'scaling'] = self.pv_scenario[scenario_id,
                                                               time_step, i]

        # Assign the WP
        for i, idx in enumerate(self.wp_list):
            self.network.sgen.at[idx,
                                 'scaling'] = self.wp_scenario[scenario_id,
                                                               time_step, i]

        # Assign the vulnerable lines
        for i, idx in enumerate(self.vulne_line_list):
            self.network.line.at[idx, 'in_service'] = bool(
                np.around(self.disruption_scenario[scenario_id, time_step, i]))

        return self.load_scenario[scenario_id, time_step], self.pv_scenario[scenario_id, time_step, :].tolist(), \
            self.wp_scenario[scenario_id, time_step, :].tolist(), self.disruption_scenario[scenario_id, time_step, :].tolist()

    def switch_action(self, input_action):
        '''
        Assign the switch action to the network
        Input: np.array of 0/1 (0: open, 1: close) with the length of num_switch
        '''
        for i, a in enumerate(input_action):
            self.network.switch.at[i, 'closed'] = bool(a)

        pass
    
    def force_line_status(self, line_status, scenario_id=None, time_step=None):
        '''
        * For evaluation MP-based methods *
        Force the line status
        line_status: a list of 0/1 (0: open, 1: close) with the length of num_line
        '''
        if scenario_id is None or time_step is None:
            for i, a in enumerate(line_status):
                self.network.line.at[i, 'in_service'] = bool(a)
        elif scenario_id is not None and time_step is not None:
            for i, a in enumerate(line_status[:,scenario_id,time_step]):
                self.network.line.at[i, 'in_service'] = self.network.line.at[i, 'in_service'] and bool(a)
        else:
            raise ValueError('The scenario_id and time_step must be provided.')
        
    def force_switch_status(self, switch_status, scenario_id=None, time_step=None):
        '''
        * For evaluation MP-based methods *
        Force the switch status
        switch_status: a list of 0/1 (0: open, 1: close) with the length of num_switch
        '''
        if scenario_id is None or time_step is None:
            for i, a in enumerate(switch_status):
                self.network.switch.at[i, 'closed'] = bool(a)
        elif scenario_id is not None and time_step is not None:
            for i, a in enumerate(switch_status[:,scenario_id,time_step]):
                self.network.switch.at[i, 'closed'] = bool(a)
        else:
            raise ValueError('The scenario_id and time_step must be provided.')

    def plot_network(self,
                     closed_line_color='black',
                     closed_line_linestyle='-',
                     closed_line_size=2,
                     opened_line_color='grey',
                     opened_line_linestyle='--',
                     opened_line_size=2,
                     critical_bus_list = [0,2,4,6],
                     critical_bus_color='red',
                     critical_bus_size=0.1,
                     critical_bus_edge_size = 2,
                     normal_bus_color='#535565',
                     normal_bus_size=0.1,
                     normal_bus_edge_size = 2,
                     sgen_size = 0.15,
                     sgen_label_xShift = 0.1,
                     sgen_label_yShift = -0.5,
                     bus_number_xShift = 0,
                     bus_number_yShift = 0.15,
                     bus_number_fontsize = 0.2,
                     switch_size = 0.1,
                     switch_distance = 0.5,
                     switch_color = 'blue',
                     saving_format=None,
                     ):
        '''
        Plot the network
        # for 7-bus system, the critical buses are bus 0, 2, 4, 6
        # for 123-bus system, the critical buses are bus [0,6,8,9,10,18,19,27,28,32,\
            34,36,41,44,45,46,47,48,50,51,52,54,59,62,64,67,68,69,70,75,78,81,87,93,\
                97,108,110,111,112,113]
        
        '''
        fig, ax = plt.subplots(dpi=300,figsize=(10,8))
        f1 = fm.FontProperties('times new roman', style='normal', size=bus_number_fontsize)
        collections = []
        bsc = plot.create_line_switch_collection(self.network, size=switch_size,distance_to_bus=switch_distance,
                                                 use_line_geodata=True,zorder=1,color=switch_color)
        collections.append(bsc)
        line_table,_,_ = self.get_line_states()
        closed_lines = self.network.line[line_table['line_connectivity']==True].index
        opened_lines = self.network.line[line_table['line_connectivity']==False].index
        closed_lc = plot.create_line_collection(self.network, closed_lines, color=closed_line_color,
                                                linewidth=closed_line_size,linestyle=closed_line_linestyle,
                                                zorder=1,use_bus_geodata=True) #create lines
        collections.append(closed_lc)
        opened_lc = plot.create_line_collection(self.network, opened_lines, color=opened_line_color,
                                                linewidth=opened_line_size,linestyle=opened_line_linestyle,
                                                zorder=1,use_bus_geodata=True) #create lines
        collections.append(opened_lc)
        critical_node = critical_bus_list
        normal_node = [i for i in range(0,self.num_bus) if i not in critical_node]
        bc = plot.create_bus_collection(self.network, normal_node, patch_type='circle', size=normal_bus_size, edgecolor='black',
                                        facecolor=normal_bus_color, linewidth=normal_bus_edge_size,zorder=2) #create buses
        collections.append(bc)
        cl_c = plot.create_bus_collection(self.network, critical_node, patch_type='circle', size=critical_bus_size, edgecolor='black',
                                        facecolor=critical_bus_color , linewidth=critical_bus_edge_size,zorder=2)
        collections.append(cl_c)
        dg_c = plot.create_sgen_collection(self.network, [i for i in range(0,self.num_sgen)], size=sgen_size,zorder=2)
        collections.append(dg_c)

        
        # plot.draw_collections([lc, bc]) # plot lines and buses
        buses = self.network.bus.index.tolist() # list of all bus indices
        coords = zip(self.network.bus_geodata.x.loc[buses].values+bus_number_xShift
                     , self.network.bus_geodata.y.loc[buses].values+bus_number_yShift) # tuples of all bus coords
        dg_bused = self.network.sgen.bus.values.tolist()
        dg_coords = zip(self.network.bus_geodata.x.loc[dg_bused].values+sgen_label_xShift
                     , self.network.bus_geodata.y.loc[dg_bused].values+sgen_label_yShift) # tuples of all bus coords
        no_bus = buses.copy()
        no_bus = [x + 1 for x in no_bus]
        bic = plot.create_annotation_collection(size=bus_number_fontsize, texts=np.char.mod('%d', no_bus), coords=coords, 
                                                prop=f1, color='black',zorder=3) # create bus indices
        collections.append(bic)
        dgic = plot.create_annotation_collection(size=bus_number_fontsize, texts=self.network.sgen.name.tolist(), coords=dg_coords, 
                                                prop=f1, color='green',zorder=3)
        collections.append(dgic)
        
        plot.draw_collections(collections,ax=ax) # plot lines, buses and bus indices
        foo_fig = plt.gcf() # 'get current figure'
        if saving_format is not None:
            foo_fig.savefig('../results/plot_network.'+saving_format, format=saving_format, dpi=300) #FIXME: CAN NOT save as svg/eps
        plt.show()
        pass

    def update_last_switch_states(self, mode='push'):
        '''
        Update the last switch states
        push: PUSH the current switch states to the last switch states
        pull: PULL BACK the last switch states to overwrite the switch states of the pandapower model
        '''
        if mode == 'push':
            self.last_switch_states = self.network.switch.copy()['closed']
        elif mode == 'pull':
            self.network.switch['closed'] = self.last_switch_states.copy()
        else:
            raise Exception("Mode must be 'push' or 'pull'.")

        pass

    def run_pf(self):
        '''
        Run the power flow
        '''
        pp.runpp(self.network)
        # Check the convergence
        if not self.network.converged:
            raise ValueError('The power flow does not converge.')

        # Return the results
        res_pmw = self.network.res_bus['p_mw']
        res_qmvar = self.network.res_bus['q_mvar']

        return res_pmw, res_qmvar

    def get_profiles(self):
        '''
        Get the current profiles of the loads, PVs, WPs, and vulnerable lines
        '''
        current_load = self.network.load['scaling']
        current_sgen = self.network.sgen['scaling']
        current_disruption = self.network.line['in_service']

        return current_load, current_sgen, current_disruption

    def get_line_states(self):
        '''
        Get the topology of the network
        '''
        # Aim to get the line connectivity and the illegal actions on the lines

        # Create a table to store the line states
        line_table = pd.DataFrame(columns=[
            'index', 'line_in_service', 'switch_closed', 'line_connectivity',
            'switch_illegal_action'
        ])
        for i in range(self.num_line):
            line_table.loc[i] = [i, True, True, True, False]
        line_table['index'] = line_table['index'].astype(int)
        # Load line states
        line_table['line_in_service'] = self.network.line['in_service']
        # Load switch states
        for index, row in self.network.switch.iterrows():
            line_table.loc[line_table['index'] == row['element'],
                           'switch_closed'] = row['closed']

        # Calculate the line connectivity
        line_table['line_connectivity'] = line_table[
            'line_in_service'] & line_table['switch_closed']

        # Calculate the illegal actions
        switch_state = self.network.switch.copy()
        # Create new columns to check the actions on the switches belonging to the out-of-service lines
        switch_state['last_switch_state'] = self.last_switch_states
        switch_state['line_in_service'] = True
        for index, row in line_table.iterrows():
            switch_state.loc[switch_state['element'] == row['index'],
                             'line_in_service'] = row['line_in_service']

        # Illegal action := (line is out of service) & (switch state is changed)
        switch_state['illegal_action'] = (switch_state['line_in_service']==False) \
            & (switch_state['closed'] != switch_state['last_switch_state'])
        for index, row in switch_state.iterrows():
            line_table.loc[line_table['index'] == row['element'],
                           'switch_illegal_action'] = row['illegal_action']
        exists_illegal_switch = any(line_table['switch_illegal_action'])
        illegal_switch_idx = line_table.index[
            line_table['switch_illegal_action'] == True].tolist()

        return line_table, exists_illegal_switch, illegal_switch_idx

    def get_load_energized_states(self):
        '''
        Check whether loads are energized by generators
        '''
        # Convert the network to a graph
        current_topology = pp.topology.create_nxgraph(self.network)

        # Create a matrix to store the connection between loads and generators
        connection_matrix = np.zeros((self.num_load, self.num_sgen))
        for i, node in enumerate(self.load_bus_list):
            res_list = set(
                pp.topology.connected_component(
                    current_topology,
                    node))  # Obtain the connected nodes for the load
            for j, connected_node in enumerate(self.sgen_bus_list):
                if connected_node in res_list:
                    connection_matrix[i][j] = 1

        # Check whether the load is connected to the generator
        load_energized_failed = (connection_matrix.sum(axis=1) == 0).any()

        return connection_matrix, load_energized_failed

    def get_v_violation(self):
        '''
        Get the voltage violation of the buses
        '''
        v_violation = [0] * self.num_bus
        V_list = self.network.res_bus['vm_pu'].tolist()
        # Handle the NaN values in the isolated buses
        V_list = [1.0 if np.isnan(x) else x for x in V_list]
        for i, value in enumerate(V_list):
            if value >= self.v_max:
                v_violation[i] = value - self.v_max
            elif value <= self.v_min:
                v_violation[i] = self.v_min - value

        return v_violation

    def get_pq_loss(self):
        '''
        Get the active and reactive power loss of the lines
        '''
        p_loss = self.network.res_line['pl_mw'].tolist()
        q_loss = self.network.res_line['ql_mvar'].tolist()

        return p_loss, q_loss

    def get_pq_unbalance(self):
        '''
        Get the active and reactive power unbalance in the system
        It checks the unbalance of the slack buses
        '''
        total_p_unbalance = self.network.res_ext_grid['p_mw'].abs().sum()
        total_q_unbalance = self.network.res_ext_grid['q_mvar'].abs().sum()

        return total_p_unbalance, total_q_unbalance

    def scenario_to_dataset(self, env_id):
        '''
        Return the dataset for the given environment id list
        shape: [len(env_id), T, N(=pv_num+wp_num+num_vulne_line+1)]
        
        '''
        pv_scenarios = [self.pv_scenario[i] for i in env_id]
        wp_scenarios = [self.wp_scenario[i] for i in env_id]
        disruption_scenarios = [self.disruption_scenario[i] for i in env_id]
        load_scenarios = [self.load_scenario[i] for i in env_id]
        dataset = np.concatenate(
            (pv_scenarios, wp_scenarios, disruption_scenarios,
             [load[..., np.newaxis] for load in load_scenarios]),
            axis=2)
        dataset = np.transpose(dataset, (0, 2, 1))

        return dataset

    def dataset_to_scenario(self, dataset):
        '''
        Convert the dataset to the scenarios: self.pv_scenario, self.wp_scenario, self.disruption_scenario
        then append the new scenarios to the original scenarios
        dataset shape: [B=batch, T, N(=pv_num+wp_num+num_vulne_line+1)]
        '''
        dataset = np.transpose(dataset, (0, 2, 1))
        self.pv_scenario = np.concatenate(
            (self.pv_scenario, dataset[:, :, 0:self.pv_num]), axis=0)
        self.wp_scenario = np.concatenate(
            (self.wp_scenario, dataset[:, :,
                                       self.pv_num:self.pv_num + self.wp_num]),
            axis=0)
        self.disruption_scenario = np.concatenate(
            (self.disruption_scenario, dataset[:, :,
                                               self.pv_num + self.wp_num:-1]),
            axis=0)
        self.load_scenario = np.concatenate(
            (self.load_scenario, dataset[:, :, -1]), axis=0)
