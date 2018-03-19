from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from copy import deepcopy
import warnings

warnings.filterwarnings('ignore')  # Warnings regarding dividing by zero, ignored

"""
/*******************************************************************
Utility for Interpreting Car-Following Data
 (1) a data point (all data from collected from one timeseries point)
 (2) a collection - set of data points to describe (consecutive) time series
 points.

Author: Britton Hammit
E-mail: bhammit1@gmail.com
********************************************************************/
"""


class DataPoint(object):
    """
    Car-Following data from a single time step
    """
    def __init__(self,timestamp,dV,dX,v_lead,v_foll,a_lead,a_foll):
        self.index = 7  # Length of an array with the input parameters (8 parameters = length of 7)
        self.timestamp = timestamp  # s
        self.dV = dV  # m/s
        self.dX = dX  # m
        self.v_lead = v_lead  # m/s
        self.v_foll = v_foll  # m/s
        self.a_lead = a_lead  # m/s2
        self.a_foll = a_foll  # m/s2

        self.headway = float(dX)/v_foll  # s
        try:
            self.time_to_collision = float(dX)/dV  # s
        except ZeroDivisionError:
            self.time_to_collision = np.nan

    def __iter__(self):
        """
        Basic Iterator Function
        :return: Self
        """
        return self

    def next(self):
        """
        Basic Next Function for Iterating through an Array
        :return: Next Point
        """
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.index

    def __getitem__(self,index):
        """
        Call an Item in a storage container using its' index
        :param index: Index of item located in a container
        :return: the new item
        """
        return self.data[index]

    def __str__(self):
        """
        Print out summary of this data point
        :return: Print out string called when instance is printed
        """
        return "Time: {} | dV: {} m/s | dX: {} m".format(round(self.timestamp,3),round(self.dV,3),round(self.dX,3))


class WyNdsDataPoint(DataPoint):
    """
    NDS data from a single time step - including all time series data available
    """
    def __init__(self,data):
        """
        Initialize object class - using all NDS variables & summarizing the Super DataPoint variables
        **NOTE** Exact formatting of the Wyoming NDS output file must be correct for the correct values to be
        selected for each attribute. If new NDS output file used - this must be redone.
        :param data: Single line of output from the Wyoming NDS time series file
        """
        self.index = len(data) + 11  # Additional attributes created
        self.data = data

        self.system_time_stamp = data[0]
        self.vtti_time_stamp = data[1]
        self.vtti_file_id = data[2]
        self.vtti_speed_network = data[3]  # km/hr
        self.vtti_speed_gps = data[4]  # km/hr
        self.vtti_accel_x = data[5]  # g
        self.vtti_accel_y = data[6]
        self.vtti_pedal_brake_state = data[7]
        self.vtti_pedal_gas_position = data[8]
        self.vtti_abs = data[9]
        self.vtti_traction_control_state = data[10]
        self.vtti_esc = data[11]
        self.vtti_lane_distance_off_center = data[12]
        self.vtti_left_line_right_distance = data[13]
        self.vtti_right_line_left_distance = data[14]
        self.vtti_left_marker_probability = data[15]
        self.vtti_right_marker_probability = data[16]
        self.vtti_light_level = data[17]
        self.vtti_gyro_z = data[18]
        self.vtti_wiper = data[19]
        self.vtti_latitude = data[20]
        self.vtti_longitude = data[21]
        self.vtti_steering_angle = data[22]
        self.vtti_steering_wheel_position = data[23]
        self.vtti_turn_signal = data[24]
        self.vtti_head_confidence = data[25]
        self.vtti_head_position_x = data[26]
        self.vtti_head_position_x_baseline = data[27]
        self.vtti_head_position_y = data[28]
        self.vtti_head_position_y_baseline = data[29]
        self.vtti_head_position_z = data[30]
        self.vtti_head_position_z_baseline = data[31]
        self.vtti_head_rotation_x = data[32]
        self.vtti_head_rotation_x_baseline = data[33]
        self.vtti_head_rotation_y = data[34]
        self.vtti_head_rotation_y_baseline = data[35]
        self.vtti_head_rotation_z = data[36]
        self.vtti_head_rotation_z_baseline = data[37]
        self.computed_time_bin = data[38]
        self.computed_day_of_month = data[39]
        self.vtti_month_gps = data[40]
        self.vtti_year_gps = data[41]
        self.vtti_eye_glance_location = data[42]
        self.vtti_alcohol_interior = data[43]
        self.vtti_airbag_driver = data[44]
        self.vtti_engine_rpm_instant = data[45]
        self.vtti_odometer = data[46]
        self.vtti_prndl = data[47]
        self.vtti_seatbelt_driver = data[48]
        self.vtti_temperature_interior = data[49]
        self.vtti_heading_gps = data[50]
        self.vtti_headlight = data[51]
        self.vtti_lane_width = data[52]
        self.vtti_object_id_t0 = data[53]
        self.vtti_object_id_t1 = data[54]
        self.vtti_object_id_t2 = data[55]
        self.vtti_object_id_t3 = data[56]
        self.vtti_object_id_t4 = data[57]
        self.vtti_object_id_t5 = data[58]
        self.vtti_object_id_t6 = data[59]
        self.vtti_object_id_t7 = data[60]
        self.vtti_range_rate_x_t0 = data[61]
        self.vtti_range_rate_x_t1 = data[62]
        self.vtti_range_rate_x_t2 = data[63]
        self.vtti_range_rate_x_t3 = data[64]
        self.vtti_range_rate_x_t4 = data[65]
        self.vtti_range_rate_x_t5 = data[66]
        self.vtti_range_rate_x_t6 = data[67]
        self.vtti_range_rate_x_t7 = data[68]
        self.vtti_range_rate_y_t0 = data[69]
        self.vtti_range_rate_y_t1 = data[70]
        self.vtti_range_rate_y_t2 = data[71]
        self.vtti_range_rate_y_t3 = data[72]
        self.vtti_range_rate_y_t4 = data[73]
        self.vtti_range_rate_y_t5 = data[74]
        self.vtti_range_rate_y_t6 = data[75]
        self.vtti_range_rate_y_t7 = data[76]
        self.vtti_range_x_t0 = data[77]
        self.vtti_range_x_t1 = data[78]
        self.vtti_range_x_t2 = data[79]
        self.vtti_range_x_t3 = data[80]
        self.vtti_range_x_t4 = data[81]
        self.vtti_range_x_t5 = data[82]
        self.vtti_range_x_t6 = data[83]
        self.vtti_range_x_t7 = data[84]
        self.vtti_range_y_t0 = data[85]
        self.vtti_range_y_t1 = data[86]
        self.vtti_range_y_t2 = data[87]
        self.vtti_range_y_t3 = data[88]
        self.vtti_range_y_t4 = data[89]
        self.vtti_range_y_t5 = data[90]
        self.vtti_range_y_t6 = data[91]
        self.vtti_range_y_t7 = data[92]
        self.vtti_headway_to_lead_vehicle = data[93]
        self.vtti_video_frame = data[94]
        self.track1_target_travel_direction = data[95]
        self.track2_target_travel_direction = data[96]
        self.track3_target_travel_direction = data[97]
        self.track4_target_travel_direction = data[98]
        self.track5_target_travel_direction = data[99]
        self.track6_target_travel_direction = data[100]
        self.track7_target_travel_direction = data[101]
        self.track8_target_travel_direction = data[102]
        self.track1_x_acc_estimated = data[103]  #m/s2
        self.track2_x_acc_estimated = data[104]
        self.track3_x_acc_estimated = data[105]
        self.track4_x_acc_estimated = data[106]
        self.track5_x_acc_estimated = data[107]
        self.track6_x_acc_estimated = data[108]
        self.track7_x_acc_estimated = data[109]
        self.track8_x_acc_estimated = data[110]
        self.track1_headway = data[111]
        self.track2_headway = data[112]
        self.track3_headway = data[113]
        self.track4_headway = data[114]
        self.track5_headway = data[115]
        self.track6_headway = data[116]
        self.track7_headway = data[117]
        self.track8_headway = data[118]
        self.track1_lane = data[119]
        self.track2_lane = data[120]
        self.track3_lane = data[121]
        self.track4_lane = data[122]
        self.track5_lane = data[123]
        self.track6_lane = data[124]
        self.track7_lane = data[125]
        self.track8_lane = data[126]
        self.track1_is_lead_vehicle = data[127]
        self.track2_is_lead_vehicle = data[128]
        self.track3_is_lead_vehicle = data[129]
        self.track4_is_lead_vehicle = data[130]
        self.track5_is_lead_vehicle = data[131]
        self.track6_is_lead_vehicle = data[132]
        self.track7_is_lead_vehicle = data[133]
        self.track8_is_lead_vehicle = data[134]
        self.track1_x_pos_processed = data[135]  # m
        self.track2_x_pos_processed = data[136]
        self.track3_x_pos_processed = data[137]
        self.track4_x_pos_processed = data[138]
        self.track5_x_pos_processed = data[139]
        self.track6_x_pos_processed = data[140]
        self.track7_x_pos_processed = data[141]
        self.track8_x_pos_processed = data[142]
        self.track1_y_pos_processed = data[143]
        self.track2_y_pos_processed = data[144]
        self.track3_y_pos_processed = data[145]
        self.track4_y_pos_processed = data[146]
        self.track5_y_pos_processed = data[147]
        self.track6_y_pos_processed = data[148]
        self.track7_y_pos_processed = data[149]
        self.track8_y_pos_processed = data[150]
        self.track1_target_id = data[151]
        self.track2_target_id = data[152]
        self.track3_target_id = data[153]
        self.track4_target_id = data[154]
        self.track5_target_id = data[155]
        self.track6_target_id = data[156]
        self.track7_target_id = data[157]
        self.track8_target_id = data[158]
        # To be added from Corresponding STAC data - Must have STAC data for this!
        self.track1_x_vel_processed = np.nan
        self.track2_x_vel_processed = np.nan
        self.track3_x_vel_processed = np.nan
        self.track4_x_vel_processed = np.nan
        self.track5_x_vel_processed = np.nan
        self.track6_x_vel_processed = np.nan
        self.track7_x_vel_processed = np.nan
        self.track8_x_vel_processed = np.nan
        # Added for car-following
        self.stac_data_available = False
        self.headway = self.headway_calc()
        self.car_following_status = self.is_car_following()
        self.lead_target_id = self.lead_target_id()
        self.time_to_collision = self.time_to_collision_calc()

        # Convert speed into m/s from km/hr
        super(WyNdsDataPoint,self).__init__(timestamp=self.vtti_time_stamp,
                                          dV=self.lead_relative_velocity(),
                                          dX=self.lead_target_dist(),
                                          v_lead=self.lead_vehicle_speed(),
                                          v_foll=self.vtti_speed_network*1000/3600,
                                          a_lead=self.lead_target_acc(),
                                          a_foll=self.vtti_accel_x*9.81)

    def is_car_following(self):
        """
        Method used to determine if a vehicle is in "car-following" by detecting if a lead vehicle exists in any track
        :return: Returns True if vehicle is in car-following and False if vehicle is not in
        """
        if np.isnan(self.track1_target_id) != True and self.track1_is_lead_vehicle == 1:
            return True
        elif np.isnan(self.track2_target_id) != True and self.track2_is_lead_vehicle == 1:
            return True
        elif np.isnan(self.track3_target_id) != True and self.track3_is_lead_vehicle == 1:
            return True
        elif np.isnan(self.track4_target_id) != True and self.track4_is_lead_vehicle == 1:
            return True
        elif np.isnan(self.track5_target_id) != True and self.track5_is_lead_vehicle == 1:
            return True
        elif np.isnan(self.track6_target_id) != True and self.track6_is_lead_vehicle == 1:
            return True
        elif np.isnan(self.track7_target_id) != True and self.track7_is_lead_vehicle == 1:
            return True
        elif np.isnan(self.track8_target_id) != True and self.track8_is_lead_vehicle == 1:
            return True
        else:
            return False

    def current_targets(self):
        """
        :return: Returns the list of all targets detected; if no targets detected, empty list is returned
        """
        targets = list()
        if np.isnan(self.track1_target_id) != True:
            targets.append(int(self.track1_target_id))
        if np.isnan(self.track2_target_id) != True:
            targets.append(int(self.track2_target_id))
        if np.isnan(self.track3_target_id) != True:
            targets.append(int(self.track3_target_id))
        if np.isnan(self.track4_target_id) != True:
            targets.append(int(self.track4_target_id))
        if np.isnan(self.track5_target_id) != True:
            targets.append(int(self.track5_target_id))
        if np.isnan(self.track6_target_id) != True:
            targets.append(int(self.track6_target_id))
        if np.isnan(self.track7_target_id) != True:
            targets.append(int(self.track7_target_id))
        if np.isnan(self.track8_target_id) != True:
            targets.append(int(self.track8_target_id))
        return targets

    def lead_target_id(self):
        """
        :return: Returns the lead vehicle target id; if no lead vehicle - returns 'nan'
        """
        if self.is_car_following() is True:
            if np.isnan(self.track1_target_id) != True and self.track1_is_lead_vehicle == 1:
                return int(self.track1_target_id)
            elif np.isnan(self.track2_target_id) != True and self.track2_is_lead_vehicle == 1:
                return int(self.track2_target_id)
            elif np.isnan(self.track3_target_id) != True and self.track3_is_lead_vehicle == 1:
                return int(self.track3_target_id)
            elif np.isnan(self.track4_target_id) != True and self.track4_is_lead_vehicle == 1:
                return int(self.track4_target_id)
            elif np.isnan(self.track5_target_id) != True and self.track5_is_lead_vehicle == 1:
                return int(self.track5_target_id)
            elif np.isnan(self.track6_target_id) != True and self.track6_is_lead_vehicle == 1:
                return int(self.track6_target_id)
            elif np.isnan(self.track7_target_id) != True and self.track7_is_lead_vehicle == 1:
                return int(self.track7_target_id)
            elif np.isnan(self.track8_target_id) != True and self.track8_is_lead_vehicle == 1:
                return int(self.track8_target_id)
            else:
                return np.nan
        else:
            return np.nan

    def lead_target_track(self):
        """
        :return: Returns the lead vehicle target track; if no lead vehicle - returns "nan"
        """
        if self.is_car_following() is True:
            if np.isnan(self.track1_target_id) != True and self.track1_is_lead_vehicle == 1:
                return 1
            elif np.isnan(self.track2_target_id) != True and self.track2_is_lead_vehicle == 1:
                return 2
            elif np.isnan(self.track3_target_id) != True and self.track3_is_lead_vehicle == 1:
                return 3
            elif np.isnan(self.track4_target_id) != True and self.track4_is_lead_vehicle == 1:
                return 4
            elif np.isnan(self.track5_target_id) != True and self.track5_is_lead_vehicle == 1:
                return 5
            elif np.isnan(self.track6_target_id) != True and self.track6_is_lead_vehicle == 1:
                return 6
            elif np.isnan(self.track7_target_id) != True and self.track7_is_lead_vehicle == 1:
                return 7
            elif np.isnan(self.track8_target_id) != True and self.track8_is_lead_vehicle == 1:
                return 8
            else:
                return np.nan
        else:
            return np.nan

    def lead_target_dist(self):
        """
        Following distance measured from radar unit (front bumper to rear bumper of lead vehicle)
        :return: Returns the distance to the lead vehicle - if no lead vehicle - returns "nan"
        """
        if self.is_car_following() is True:
            if np.isnan(self.track1_target_id) != True and self.track1_is_lead_vehicle == 1:
                return self.track1_x_pos_processed
            elif np.isnan(self.track2_target_id) != True and self.track2_is_lead_vehicle == 1:
                return self.track2_x_pos_processed
            elif np.isnan(self.track3_target_id) != True and self.track3_is_lead_vehicle == 1:
                return self.track3_x_pos_processed
            elif np.isnan(self.track4_target_id) != True and self.track4_is_lead_vehicle == 1:
                return self.track4_x_pos_processed
            elif np.isnan(self.track5_target_id) != True and self.track5_is_lead_vehicle == 1:
                return self.track5_x_pos_processed
            elif np.isnan(self.track6_target_id) != True and self.track6_is_lead_vehicle == 1:
                return self.track6_x_pos_processed
            elif np.isnan(self.track7_target_id) != True and self.track7_is_lead_vehicle == 1:
                return self.track7_x_pos_processed
            elif np.isnan(self.track8_target_id) != True and self.track8_is_lead_vehicle == 1:
                return self.track8_x_pos_processed
            else:
                return np.nan
        else:
            return np.nan

    def headway_calc(self):
        """
        Headway: following distance/ following vehicle velocity
        :return: Returns the time headway to the lead vehicle; if no lead vehicle - returns "nan"
        """
        if self.is_car_following() is True:
            headway = self.lead_target_dist()/(self.vtti_speed_network*1000/3600)
        else:
            headway = np.nan
        return headway

    def time_to_collision_calc(self):
        if self.is_car_following() is True:
            dX = self.lead_target_dist()
            dV = self.lead_relative_velocity()
            ttc = float(dX)/dV
        else:
            ttc = np.nan
        return ttc

    def lead_target_acc(self):
        """
        Acceleration estimated from the radar computations made by VTTI
        :return: Returns the lead vehicle acceleration; if no lead vehicle - returns "nan"
        """
        if self.is_car_following() is True:
            if np.isnan(self.track1_target_id) != True and self.track1_is_lead_vehicle == 1:
                return self.track1_x_acc_estimated
            elif np.isnan(self.track2_target_id) != True and self.track2_is_lead_vehicle == 1:
                return self.track2_x_acc_estimated
            elif np.isnan(self.track3_target_id) != True and self.track3_is_lead_vehicle == 1:
                return self.track3_x_acc_estimated
            elif np.isnan(self.track4_target_id) != True and self.track4_is_lead_vehicle == 1:
                return self.track4_x_acc_estimated
            elif np.isnan(self.track5_target_id) != True and self.track5_is_lead_vehicle == 1:
                return self.track5_x_acc_estimated
            elif np.isnan(self.track6_target_id) != True and self.track6_is_lead_vehicle == 1:
                return self.track6_x_acc_estimated
            elif np.isnan(self.track7_target_id) != True and self.track7_is_lead_vehicle == 1:
                return self.track7_x_acc_estimated
            elif np.isnan(self.track8_target_id) != True and self.track8_is_lead_vehicle == 1:
                return self.track8_x_acc_estimated
            else:
                return np.nan
        else:
            return np.nan

    def lead_relative_velocity(self):
        """
        The relative velocity is measured from the radar unit on the NDS Vehicle
        In Wyoming's Approach 2 data request, the X_VEL_PROCESSED data field was left out of the request,
        therefore, the relative velocity could not be directly gathered from the data. FHWA's STAC provided
        means to access this variable - and a portion of the trips were queried (those available under FHWA's
        DUL) and STAC radar data was gathered. For those trips we are able to directly pull relative velocity
        data -- for other trips we need to compute relative velocity in a subsequent step.

        Relative Velocity is defined as: dV = v_foll - v_lead

        :return: returns relative velocity if vehicle is in car-following and STAC data is available,
                 returns NINF if STAC data is available and vehicle is not in car-following
                 returns NAN if no STAC data is available
        """
        if self.stac_data_available is True:
            if self.is_car_following() is True:
                if np.isnan(self.track1_target_id) == False and self.track1_is_lead_vehicle == 1:
                    return -self.track1_x_vel_processed
                elif np.isnan(self.track2_target_id) == False and self.track2_is_lead_vehicle == 1:
                    return -self.track2_x_vel_processed
                elif np.isnan(self.track3_target_id) == False and self.track3_is_lead_vehicle == 1:
                    return -self.track3_x_vel_processed
                elif np.isnan(self.track4_target_id) == False and self.track4_is_lead_vehicle == 1:
                    return -self.track4_x_vel_processed
                elif np.isnan(self.track5_target_id) == False and self.track5_is_lead_vehicle == 1:
                    return -self.track5_x_vel_processed
                elif np.isnan(self.track6_target_id) == False and self.track6_is_lead_vehicle == 1:
                    return -self.track6_x_vel_processed
                elif np.isnan(self.track7_target_id) == False and self.track7_is_lead_vehicle == 1:
                    return -self.track7_x_vel_processed
                elif np.isnan(self.track8_target_id) == False and self.track8_is_lead_vehicle == 1:
                    return -self.track8_x_vel_processed
            elif self.is_car_following() is False:
                return np.NINF
        elif self.stac_data_available is False:
            return np.nan

    def lead_vehicle_speed(self,stac=True):
        """
        Method returns the lead vehicle speed [m/s]
        :param stac: [True or False Boolean]
        Necessary for Moving Average Filter Function - without stac data, when dV is calculated using kinematics.
        :return: lead vehicle speed [m/s]
        """
        if stac is True:
            dV = self.lead_relative_velocity()
            if np.isfinite(dV) == True:
                return self.vtti_speed_network*1000/3600 - dV  # m/s
            else:
                return np.nan
        else:
            dV = self.dV
            if np.isfinite(dV) == True:
                return self.vtti_speed_network*1000/3600 - dV  # m/s
            else:
                return np.nan

    def reset_super_attributes(self):
        """
        Method resets the instance "super" DataPoint values - this is necessary when updating the relative velocity
        values from the STAC radar data.
        :return: N/A
        """
        super(WyNdsDataPoint,self).__init__(timestamp=self.vtti_time_stamp,
                                          dV=self.lead_relative_velocity(),
                                          dX=self.lead_target_dist(),
                                          v_lead= self.lead_vehicle_speed(),
                                          v_foll=self.vtti_speed_network*1000/3600,
                                          a_lead=self.lead_target_acc(),
                                          a_foll=self.vtti_accel_x*9.81)


class WyNdsReducedDataPoint(DataPoint):
    """
    Import the reduced data point for computational efficiency - This should be a single CF Event!
    """
    def __init__(self,data):
        self.index = len(data)
        self.timestamp = int(float(data[0]))  # VTTI Timestamp
        self.dV = float(data[1])  # m/s
        self.dX = float(data[2])  # m
        self.v_lead = float(data[3])  # m/s
        self.v_foll = float(data[4])  # m/s
        self.a_lead = float(data[5])  # m/s2
        self.a_foll = float(data[6])  # m/s2
        self.headway = float(data[7])  # s
        self.ttc = float(data[8])  # s
        self.latitude = float(data[9])
        self.longitude = float(data[10])
        self.z_gyro = float(data[11])
        self.lane_dist_off_center = float(data[12])
        self.wiper = float(data[13])
        self.cf_status = data[14]  # T/F
        super(WyNdsReducedDataPoint, self).__init__(timestamp=self.timestamp,
                                                       dV=self.dV,
                                                       dX=self.dX,
                                                       v_lead=self.v_lead,
                                                       v_foll=self.v_foll,
                                                       a_lead=self.a_lead,
                                                       a_foll=self.a_foll)


class WzDataPoint(DataPoint):
    """
    NDS data from a single time step - including all time series data available
    """
    def __init__(self, data):
        self.index = len(data)
        self.driver_id = data[0]
        self.unique_inst = data[1]
        self.instance_id = data[2]
        self.timestamp = data[3]
        self.leader_vel = data[4]
        self.leader_accel = data[5]
        self.follower_vel = data[6]
        self.follower_accel = data[7]
        self.delta_dist = data[8]
        self.delta_vel = data[9]
        self.bad_flag = data[10]
        super(WzDataPoint, self).__init__(timestamp=self.timestamp,
                                             dV=self.delta_vel,
                                             dX=self.delta_dist,
                                             v_lead=self.leader_vel,
                                             v_foll=self.follower_vel,
                                             a_lead=self.leader_accel,
                                             a_foll=self.follower_accel)


class VissimSyntheticDataPoint(DataPoint):
    """
    Data point derived from vissim simulation
    """
    def __init__(self, data):
        self.index = len(data)
        self.vehicle_no = data[0]
        self.desired_velocity = data[1]
        self.timestamp = data[2]  # sec
        self.dV = data[3]  # m/s
        self.dX = data[4]  # m
        self.v_lead = data[5]  # m/s
        self.v_foll = data[6]  # m/s
        self.a_lead = data[7]  # m/s2
        self.a_foll = data[8]  # m/s2
        super(VissimSyntheticDataPoint,self).__init__(timestamp=self.timestamp,
                                                      dV=self.dV,
                                                      dX=self.dX,
                                                      v_lead=self.v_lead,
                                                      v_foll=self.v_foll,
                                                      a_lead=self.a_lead,
                                                      a_foll=self.a_foll)


class PointCollection:
    """
    Collection of Data Points - list of DataPoints
    """
    def __init__(self,list_of_data_points = None):
        """
        Initialise the PointCollection - either with an existing list of data points, or as an empty object
        to be appended to later.
        :param list_of_data_points: Initial List of data points, or an empty point collection can be created
        """

        self.index = 0  # Length of an array with the input parameters (8 parameters = length of 7)
        self.list_of_data_points = list()
        self.timestamp = list()  # s
        self.dV = list()  # m/s
        self.dX = list()  # m
        self.v_lead = list()  # m/s
        self.v_foll = list()  # m/s
        self.a_lead = list()  # m/s2
        self.a_foll = list()  # m/s2
        self.headway = list()  # s
        if list_of_data_points is None:
            pass
        else:
            # Append values from DataPoint to PointCollection Attributes
            flag = False  # Flag variable assigned to indicate if the dV and V_lead need to be calculated
            for i in range(len(list_of_data_points)):
                self.index += 1
                self.list_of_data_points.append(list_of_data_points[i])
                self.timestamp.append(list_of_data_points[i].timestamp)
                self.dX.append(list_of_data_points[i].dX)
                self.v_foll.append(list_of_data_points[i].v_foll)  # m/s
                self.a_lead.append(list_of_data_points[i].a_lead)
                self.a_foll.append(list_of_data_points[i].a_foll)
                self.headway.append(list_of_data_points[i].headway)
                if isinstance(list_of_data_points[i],WyNdsDataPoint) is not True:
                    self.dV.append(list_of_data_points[i].dV)
                    self.v_lead.append(list_of_data_points[i].v_lead)
                else:
                    if list_of_data_points[i].stac_data_available is True:
                        self.dV.append(list_of_data_points[i].dV)
                        self.v_lead.append(list_of_data_points[i].v_lead)
                    else:
                        flag = True
            if flag is True:
                # Kinematics approach to calculate the dV and v_lead when data is not available for WY NDS data
                # Calculated using corresponding data collected
                self.dV.append(np.nan)
                self.v_lead.append(np.nan)
                for i in range(len(list_of_data_points)-1):
                    # Calculate Time interval (vtti time steps at 1000 increments)
                    dt = (list_of_data_points[i+1].timestamp - list_of_data_points[i].timestamp)/1000  # s
                    # Calculate distance traveled by the following vehicle
                    d_trav_foll = list_of_data_points[i].v_foll * dt  # m
                    # Calculate distance traveled by the lead vehicle
                    d_trav_lead = list_of_data_points[i+1].dX + d_trav_foll-list_of_data_points[i].dX  # m
                    # Calculate the target (lead vehicle) velocity
                    v_target = d_trav_lead / dt - list_of_data_points[i+1].a_lead * dt
                    # Append to PointCollection Lists:
                    self.v_lead.append(v_target)  # m/s
                    self[i].v_lead = v_target
                    self.dV.append(list_of_data_points[i].v_foll - v_target)  # m/s
                    self[i].dV = list_of_data_points[i].v_foll - v_target
                self.time_to_collision()  # Update Data Point Time to Collision Values.

    def __iter__(self):
        return self

    def next(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.list_of_data_points[self.index]

    def __getitem__(self,index):
        return self.list_of_data_points[index]

    def point_append(self,list_of_data_points):
        """
        Append a List of DataPoints to an Existing PointCollection
        !! Important - This is assuming that we are importing from an existing point collection
        in which the missing WyNDS radar data has already been accounted for
        :param list_of_data_points: List of DataPoint Objects
        :return: None
        """
        # Append values from DataPoint to PointCollection Attributes
        if isinstance(list_of_data_points,list) is True:
            for i in range(len(list_of_data_points)):
                self.index += 1
                self.list_of_data_points.append(list_of_data_points[i])
                self.timestamp.append(list_of_data_points[i].timestamp)
                self.dX.append(list_of_data_points[i].dX)
                self.v_foll.append(list_of_data_points[i].v_foll)
                self.a_lead.append(list_of_data_points[i].a_lead)
                self.a_foll.append(list_of_data_points[i].a_foll)
                self.headway.append(list_of_data_points[i].headway)
                self.v_lead.append(list_of_data_points[i].v_lead)
                self.dV.append(list_of_data_points[i].dV)
        else:
            self.index += 1
            self.list_of_data_points.append(list_of_data_points)
            self.timestamp.append(list_of_data_points.timestamp)
            self.dX.append(list_of_data_points.dX)
            self.v_foll.append(list_of_data_points.v_foll)
            self.a_lead.append(list_of_data_points.a_lead)
            self.a_foll.append(list_of_data_points.a_foll)
            self.headway.append(list_of_data_points.headway)
            self.v_lead.append(list_of_data_points.v_lead)
            self.dV.append(list_of_data_points.dV)

    def reset_attributes(self):
        list_of_data_points = deepcopy(self.list_of_data_points)

        # Reset Attributes
        self.index = 0  # Length of an array with the input parameters (8 parameters = length of 7)
        self.list_of_data_points = list()
        self.timestamp = list()  # s
        self.dV = list()  # m/s
        self.dX = list()  # m
        self.v_lead = list()  # m/s
        self.v_foll = list()  # m/s
        self.a_lead = list()  # m/s2
        self.a_foll = list()  # m/s2
        self.headway = list()  # s

        # Append Points
        for i in range(len(list_of_data_points)):
            self.point_append(list_of_data_points[i])

    def point_count(self):
        """
        Returns the number of DataPoints contained within the Collection
        :return: Integer Count Value
        """
        return self.index

    def time_elapsed(self):
        """
        Total amount of time in which the PointCollection covers
        Assumes a data collection frequency of 10Hz (1/10 sec)
        :return: Float Time Value [seconds]
        """
        return self.point_count()/float(10)

    def dist_traveled(self):
        """
        Compute the distance traveled by the following vehicle using the vehicle speed as a metric
        :return: Distance traveled during PointCollection [meters]
        """
        temp_dist = 0
        for i in range(self.index):
            speed_temp = self.v_foll[i]
            try:
                int(speed_temp)
                dist = speed_temp / float(36000)
                temp_dist += dist
            except ValueError:
                continue
        return temp_dist

    def start_stop_timestamp(self):
        """
        Determine the starting and stopping timestamp values
        :return: Matrix containing [start timestamp, stop timestamp]
        """
        try: start = int(self.timestamp[0])
        except ValueError:
            try: start = int(self.timestamp[1])
            except ValueError:
                try: start = int(self.timestamp[2])
                except ValueError:
                    try: start = int(self.timestamp[3])
                    except ValueError:
                        try: start = int(self.timestamp[4])
                        except ValueError:
                            raise ValueError('Numeric Value not within first five points of collection')
        stop = int(self.timestamp[self.index-1])

        return [start,stop]

    def percent_car_following(self, max_cf_dist=60):
        """
        Method identifies the percent of the PointCollection in which a subject vehicle is in car-following
        :return: percentage of PointCollection in car-following mode [normalized percent - decimal]
        """
        temp = 0
        """
        for i in range(self.index):
            if np.isfinite(self.dV[i]) == True:
                temp+=1
        """
        # Changed method such that it will also work with synthetic and other datapoints
        for i in range(self.index):
            if self.dX[i] <= max_cf_dist:
                temp+=1
        try:
            return float(temp) / self.index
        except ZeroDivisionError:
            return np.nan

    def time_car_following(self):
        """
        Method identifies the time in car-following
        :return: Time in car-following [minutes]
        """
        return self.percent_car_following()*self.time_elapsed()/60  # min

    ### Plots
    def plot_t_X(self,trip_no=None):
        fig = plt.figure(figsize=(12,10))  # Size of figure
        if trip_no is not None:
            fig.suptitle('Following Distance over Time for {}'.format(trip_no),fontsize=18, fontweight='bold')
        else:
            fig.suptitle('Following Distance over Time', fontsize=18, fontweight='bold')
        plt.scatter(self.timestamp,self.dX,color='blue',s=1.5)
        plt.ylabel("Distance to Lead Vehicle [m]",fontsize=16)
        plt.xlabel("Timestamp", fontsize=16)
        # plt.xlabel("VTTI Timestamp [10000=1sec]",fontsize=16)
        plt.ylim([0,60])
        plt.close()
        return fig

    def plot_dV_X(self,trip_no=None):
        fig = plt.figure(figsize=(12,10))  # Size of figure
        if trip_no is not None:
            fig.suptitle('Relative Velocity - Following Distance for {}'.format(trip_no),fontsize=18, fontweight='bold')
        else:
            fig.suptitle('Relative Velocity - Following Distance', fontsize=18,fontweight='bold')
        plt.scatter(self.dV,self.dX,color='blue',s=1.5)
        plt.ylabel("Distance to Target, X [m]",fontsize=16)
        plt.xlabel("Change in Velocity, dV [m/s]",fontsize=16)
        #plt.ylim([0,60])  # Distance to lead vehicle
        # Make plot axes symmetric
        dV_abs = list()
        for i in range(len(self.dV)):
            dV_abs.append(abs(self.dV[i]))
        plt.xlim([-np.nanmax(dV_abs) - 0.5, np.nanmax(dV_abs) + 0.5])
        plt.close()
        return fig

    def plot_t_dV(self,trip_no=None):
        fig = plt.figure(figsize=(12,10))  # Size of figure
        if trip_no is not None:
            fig.suptitle('Relative Velocity over Time for {}'.format(trip_no),fontsize=18, fontweight='bold')
        else:
            fig.suptitle('Relative Velocity over Time', fontsize=18, fontweight='bold')
        #plt.scatter(self.timestamp,self.dV_collection,color='green',s=1.5)
        plt.scatter(self.timestamp,self.dV,color='blue',s=1.5)
        plt.xlabel("Timestamp", fontsize=16)
        #plt.xlabel("VTTI Timestamp [10000=1sec]",fontsize=16)
        plt.ylabel("Change in Velocity, dV [m/s]",fontsize=16)
        plt.close()
        return fig

    ### Analysis
    def time_to_collision(self):
        """
        Computes the time to collision (basic TTC assumptions assuming neither vehicle changes trajectory)
        :return: list of time to collision values for each DataPoint within the PointCollection [sec]
        """
        ttc_list = list()
        ttc_list.append(np.nan)  # In order to make it the same length as other lists within PointCollection
        dX = self.dX
        dV = self.dV
        for i in range(self.index-1):
            if dV[i] > 0:
                ttc = dX[i] / dV[i]
            else:
                ttc = np.nan
            ttc_list.append(ttc)
            del ttc
        return ttc_list

    def export_timeseries_file(self,file):
        """
        Generate time series file of a Point Collection - only including select variables.
        :param file: File ready for writing to save time series data
        :return: N/A
        """
        file.write("timestamp,dV [m/s],dX [m],v_lead [m/s],v_foll [m/s],a_lead [m/s2],a_foll [m/s2],headway [s],ttc [s]")
        file.write("\n")

        ttc_list = self.time_to_collision()
        # Write Data to File
        for i in range(self.index):
            file.write("{},{},{},{},{},{},{},{},{}".format(self.timestamp[i],
                                                            self.dV[i],
                                                            self.dX[i],
                                                            self.v_lead[i],
                                                            self.v_foll[i],
                                                            self.a_lead[i],
                                                            self.a_foll[i],
                                                            self.headway[i],
                                                            ttc_list[i]))
            file.write("\n")
        file.close()

    def cf_summary_plot(self,trip_no=None):
        """

        :param trip_no:
        :return:
        """
        # Plotting
        fig = plt.figure(figsize=(16, 12))  # Size of figure
        if trip_no is not None:
            fig.suptitle('Summary Plots for Trip {}: {} sec'.format(trip_no,self.time_elapsed()), fontsize=18, fontweight='bold')
        else:
            fig.suptitle('Summary Plots', fontsize=18, fontweight='bold')

        # Vehicle Speeds
        a = plt.subplot2grid(shape=(6,6),loc=(0,0),rowspan=2,colspan=3)
        v_foll = list()
        v_lead = list()
        """
        for i in range(len(self.v_foll)):
            v_foll.append(self.v_foll[i]*3600/1000)
            v_lead.append(self.v_lead[i]*3600/1000)
        """
        a.plot(self.timestamp,self.v_foll,color='red',label='Following Vehicle')
        a.plot(self.timestamp,self.v_lead,color='green',label='Lead Vehicle')
        a.set_xlabel('Timestamp')
        a.set_ylabel('Speed [m/s]')
        a.set_ylim(ymin=0)
        a.legend()
        a.set_title('Vehicle Speeds')

        # Vehicle Accelerations
        b = plt.subplot2grid(shape=(6,6),loc=(0,3),rowspan=2,colspan=3)
        b.plot(self.timestamp, self.a_lead, color='green', label='Lead Vehicle')
        b.plot(self.timestamp,self.a_foll,color='red',label='Following Vehicle')
        b.set_xlabel('Timestamp')
        b.set_ylabel('Acceleration [m/s2]')
        b.legend()
        b.set_title('Vehicle Accelerations')

        # Following Distance
        c = plt.subplot2grid(shape=(6,6),loc=(2,0),rowspan=2,colspan=3)
        c.plot(self.timestamp,self.dX,color='blue')
        c.set_xlabel('Timestamp')
        c.set_ylabel('Following Distance [m]')
        c.set_ylim(ymin=0,ymax=60)
        c.set_title('Following Distance')

        # Relative Velocity
        d = plt.subplot2grid(shape=(6,6),loc=(4,0),rowspan=2,colspan=3)
        d.plot(self.timestamp,self.dV,color='blue')
        d.set_xlabel('Timestamp')
        d.set_ylabel('Relative Velocity [foll-lead] [m/s]')
        d.set_title('Relative Velocity')

        # Foll Distance & Relative Velocity
        e = plt.subplot2grid(shape=(6,6),loc=(2,3),rowspan=4,colspan=3)
        e.plot(self.dV,self.dX,color='blue')
        e.set_xlabel('Relative Velocity [foll-lead] [m/s]')
        e.set_ylabel('Following Distance [m]')
        e.set_title('Psychophysical Plane')
        dV_abs = list()
        for i in range(len(self.dV)):
            dV_abs.append(abs(self.dV[i]))
        e.set_xlim([-np.nanmax(dV_abs) - 0.5, np.nanmax(dV_abs) + 0.5])
        e.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        e.set_ylim(ymin=0,ymax=60)
        fig.subplots_adjust(plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=1.5))
        plt.close()
        return fig

    ### Trajectory Analysis - Replacing Following Vehicle Parameters ###
    def new_foll_veh_collection(self,v_foll,a_foll,dX):
        """

        :param v_foll: List of new velocity values
        :param a_foll: List of new acclerations
        :param dX: List of new following distances
        :return: Collection with new following vehicle
        """
        timestamp = self.timestamp  # s
        dV = list()  # m/s - foll-lead
        dX = dX  # m
        v_lead = self.v_lead  # m/s
        v_foll = v_foll  # m/s
        a_lead = self.a_lead  # m/s2
        a_foll = a_foll  # m/s2

        for i in range(len(timestamp)):
            dV.append(v_foll[i]-v_lead[i])

        list_of_data_points = list()
        for i in range(len(timestamp)):
            list_of_data_points.append(DataPoint(timestamp[i],dV[i],dX[i],v_lead[i],v_foll[i],a_lead[i],a_foll[i]))
        new_cf_collection = PointCollection(list_of_data_points)

        return new_cf_collection


class WyNdsPointCollection(PointCollection):

    def list_of_target_ids(self):
        """
        :return: Return a list of all targets identified within the Collection
        """
        unique_target_ids = list()
        temp_targets = list()
        for i in range(len(self.list_of_data_points)):
            temp_targets.append(self.list_of_data_points[i].current_targets())
        for i in range(len(temp_targets)):
            if len(temp_targets[i]) != 0:
                for j in range(len(temp_targets[i])):
                    if temp_targets[i][j] not in unique_target_ids:
                        unique_target_ids.append(temp_targets[i][j])
        return unique_target_ids

    def list_of_lead_targets(self):
        """
        :return: Return a list of all identified LEAD targets within the Collection
        """
        unique_lead_target_ids = list()
        temp_targets = list()
        for i in range(len(self.list_of_data_points)):
            temp_targets.append(self.list_of_data_points[i].lead_target_id)
        for i in range(len(temp_targets)):
            if temp_targets[i] is not None:
                if temp_targets[i] not in unique_lead_target_ids and np.isnan(temp_targets[i]) == False:
                    unique_lead_target_ids.append(int(temp_targets[i]))
        return unique_lead_target_ids

    def start_lat_long(self):
        start_lat = np.nan
        index = 0
        while np.isnan(start_lat) == True:
            try: start_lat = self.list_of_data_points[index].vtti_latitude / 1.0
            except ValueError:
                pass
            except IndexError:
                break
            index += 1

        start_long = np.nan
        index = 0
        while np.isnan(start_long) == True:
            try: start_long = self.list_of_data_points[index].vtti_longitude / 1.0
            except ValueError:
                pass
            except IndexError:
                break
            index += 1

        return [start_lat,start_long]

    def stop_lat_long(self):
        stop_lat = np.nan
        index = len(self.list_of_data_points)-1
        while np.isnan(stop_lat) == True:
            try: stop_lat = self.list_of_data_points[index].vtti_latitude / 1.0
            except ValueError:
                pass
            except IndexError:
                break
            index -= 1

        stop_long = np.nan
        index = len(self.list_of_data_points)-1
        while np.isnan(stop_long) == True:
            try: stop_long = self.list_of_data_points[index].vtti_longitude / 1.0
            except ValueError:
                pass
            except IndexError:
                break
            index -= 1

        return [stop_lat,stop_long]

    def time_day_month_year(self):
        time = np.nan
        index = 0
        flag = False
        while np.isnan(time) == True:
            try:
                time = self.list_of_data_points[index].computed_time_bin
            except ValueError:
                pass
            except IndexError:
                flag = True
                break
            index += 1
        if flag is False:
            day = self.list_of_data_points[index-1].computed_day_of_month
            month = self.list_of_data_points[index-1].vtti_month_gps
            year = self.list_of_data_points[index-1].vtti_year_gps
        else:
            time = np.nan
            day = np.nan
            month = np.nan
            year = np.nan
        return [time,day,month,year]

    def percent_wiper_settings(self):
        wiper_setting_options = [np.nan, 0., 1., 2., 3., 254., 255.]
        wiper_setting_count = [0 for i in range(len(wiper_setting_options))]
        wiper_setting_percent = [0 for i in range(len(wiper_setting_options))]
        for i in range(len(self.list_of_data_points)):
            for j in range(len(wiper_setting_options)):
                if self.list_of_data_points[i].vtti_wiper == wiper_setting_options[j]:
                    wiper_setting_count[j] += 1

        for k in range(len(wiper_setting_count)):
            wiper_setting_percent[k] = wiper_setting_count[k] / float(self.point_count())

        return wiper_setting_percent
    
    def percent_wipers_active(self):
        temp = 0
        for i in range(len(self.list_of_data_points)):
            wiper_status = self.list_of_data_points[i].vtti_wiper
            if wiper_status == 1 or wiper_status == 2 or wiper_status == 3:
                temp += 1
        return round(float(temp)/len(self.list_of_data_points),3)

    def cf_instance_lead_target_id(self):
        temp = self.list_of_data_points[0].lead_target_id()
        for i in range(len(self.list_of_data_points)):
            if temp == self.list_of_data_points[i].lead_target_id():
                continue
            else:
                temp = False
        return temp

    def time_continuous_car_following(self):
        if self.percent_car_following() != 0:
            lead_targets = self.list_of_lead_targets()
            list_of_collections = [PointCollection() for row in range(len(lead_targets))]
            # Sort Data Points in CF behavior based on target ID into DataCollections
            for i in range(self.point_count()):
                for j in range(len(lead_targets)):
                    if self[i].lead_target_id == lead_targets[j]:
                        try:
                            if self[i].vtti_time_stamp < list_of_collections[j][
                                        list_of_collections[j].point_count() - 1].vtti_time_stamp + 3000:
                                list_of_collections[j].point_append(self[i])
                        except IndexError:
                            list_of_collections[j].point_append(self[i])
            list_length = list()
            for i in range(len(list_of_collections)):
                list_length.append(list_of_collections[i].time_elapsed())

            return np.nanmax(list_length)
        else:
            return 0

    def time_to_collision(self):
        """
        Computes the time to collision (basic TTC assumptions assuming neither vehicle changes trajectory)
        # THIS UPDATES THE TTC Values!
        :return: list of time to collision values for each DataPoint within the PointCollection [sec]
        """
        ttc_list = list()
        ttc_list.append(np.nan)  # In order to make it the same length as other lists within PointCollection
        dX = self.dX
        dV = self.dV
        for i in range(self.index - 1):
            if dV[i] > 0:
                ttc = dX[i] / dV[i]
            else:
                ttc = np.nan
            ttc_list.append(ttc)
            self.list_of_data_points[i].time_to_collision = ttc
            del ttc
        return ttc_list

    ### Analysis
    def moving_average(self, resolution='automatically_defined'):
        """

        :param resolution:
        :return:
        """
        X = self.dV

        # Assumes consecutive timestamps
        if resolution == 'automatically_defined':
            resolution = int(len(X)*0.01)

        Y = [np.nan for i in range(len(X))]

        for index in range(int(resolution/2), len(Y)-int(resolution/2)):
            Y[index] = 0
            counter = 0
            for j in range(-int(resolution/2),int(resolution/2)):
                if np.isnan(X[index-j])== False:
                    Y[index]=Y[index]+X[index-j]
                    counter += 1
            if np.isnan(Y[index]) == True:
                print 'nan'
                Y[index] = Y[index-1]
            else:
                if counter == 0:
                    Y[index] = np.nan
                else:
                    Y[index] = Y[index]/counter

        # Change the first values to be constant with the first known velocity and the last with the last known...
        # This is important for the calibration procedure... needs values.
        for index in range(0,int(resolution/2)):
            Y[index] = Y[int(resolution/2)]
        for index in range(len(Y)-int(resolution/2),len(Y)):
            Y[index] = Y[len(Y)-int(resolution/2)-1]

        # Replacing values BOTH in the DataPoint and in the PointCollection
        for i in range(len(self.list_of_data_points)):
            self.list_of_data_points[i].dV = Y[i]
            temp = self.list_of_data_points[i].lead_vehicle_speed()
            self.list_of_data_points[i].v_lead = self.list_of_data_points[i].lead_vehicle_speed(stac=False)  # Updates lead vehicle speed value considering new dV

        self.reset_attributes()  # Updates PointCollection.dV
    
    def summary_statistics(self, file, trip_no):
        """
        Generates summary file of relevant statistics:
        Network Speed, X_Accel, VTTI_Gyro, VTTI_Dist_Off_Center, VTTI_Wiper, VTTI_Headway
        :param file: Created file for holding summary statistics
        :return: N/A
        """
        warnings.filterwarnings('ignore')  # Warnings regarding all "nan" values ignored
        # List of measurements for statistics:
        measure_names = ['v_foll', 'a_foll', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway', 'dX']
        measure_values = []

        for i in range(len(measure_names)):
            temp = list()
            for j in range(self.point_count()):
                temp.append(getattr(self.list_of_data_points[j], measure_names[i]))
            measure_values.append(NewList(temp))
        del temp

        file.write("File: {}".format(trip_no))  # Print Headers to file
        file.write("\n")
        file.write("Variables,")
        for i in range(len(measure_names)):
            if i != len(measure_names) - 1:
                file.write("{},".format(measure_names[i]))
            else:
                file.write("{}".format(measure_names[i]))
        file.write("\n")

        stats_operations_names = ['Mean', 'Max', 'Min', 'Median', 'Percentile85', 'Stdev', 'Variance',
                                  'Coeff of Variation']
        operations_names = ['mean', 'maximum', 'minimum', 'median', 'percentile', 'standard_deviation', 'variance',
                            'coeff_variation']

        for i in range(len(operations_names)):
            for j in range(len(measure_values)):
                if j == 0:
                    file.write("{},".format(stats_operations_names[i]))
                temp = getattr(measure_values[j], operations_names[i])
                if j != len(measure_values) - 1:
                    file.write("{},".format(temp()))
                else:
                    file.write("{}".format(temp()))
                    file.write("\n")
        del temp
        file.close()

    def percent_available(self, attribute):
        """
        Method that calculates the percent of datapoints with a finite value available
        :param attribute: string with WyNdsDataPoint attribute name
        :return: decimal percentage
        """
        count = 0
        for i in range(self.index):
            temp = getattr(self[i], attribute)
            if np.isfinite(temp) == True:
                count += 1
        return count / float(self.point_count())

    def segmentation(self,split_length=60):
        """
        Method intended to generate a list of point collections of a specified length
        :param split_length: time period by which the segmentation of the trips will occur [seconds]
        :return: list of "Point Collections" comprising the original "self"
        """
        list_of_collections = list()
        index = 0
        while (self.point_count()-index) > split_length*10:  # Mult by 10 because 10 Hz
            temp_collection = WyNdsPointCollection()
            while temp_collection.time_elapsed() < split_length:
                temp_collection.point_append(self.list_of_data_points[index])
                index += 1
            list_of_collections.append(temp_collection)
            del temp_collection
        return list_of_collections

    def video_observation_template(self,file,trip_no,split_length=60):
        """
        Method used to generate a Manual Video Observation Template for manual viewing of the NDS Forward Video
        :param file:
        :param trip_no:
        :param split_length: length of split (segmentation) [seconds], default = 60sec
        :return:
        """

        split_point_collections = self.segmentation(split_length)

        file.write('Event ID, {}'.format(trip_no))
        file.write('\n')
        file.write('Number of Samples, {}'.format(len(split_point_collections)))
        file.write(',,,')
        file.write('\n')
        file.write('IMPORTANT - PLEASE DO NOT CHANGE ANY FORMATTING ON THIS PAGE')
        file.write('\n')
        file.write(',,,')
        file.write('\n')
        file.write('Column Sizes MAY be adjusted but all data must remain in their intended locations')
        file.write('\n')
        file.write('For best viewing: adjust columns A - F to ~20')
        file.write('\n')
        file.write('Do not add or delete columns/rows')
        file.write('\n')
        file.write('Do not erase any existing text or data')
        file.write('\n')
        file.write('ONLY add to the sheet where requested')
        file.write('\n')
        file.write('Thank you!')
        file.write('\n')
        file.write(',,,')
        file.write('\n')
        file.write(',,,')
        file.write('\n')
        file.write('Reviewer Name, TYPE NAME HERE')
        file.write('\n')
        file.write('Reviewed Date, TYPE DATE HERE IN FORMAT: Month/Day/Year')
        file.write('\n')
        file.write(',,,')
        file.write('\n')
        file.write('When Complete: Resave file as {}_C_output_complete.csv'.format(trip_no))
        file.write('\n')
        file.write(',,,')
        file.write('\n')
        file.write(',,,')
        file.write('\n')
        file.write(
            'Sample Number, Elapsed Time (min),Start Timestamp, Stop Timestamp, Is Freeway?, Weather Condition, Surface Condition, Visibility, Traffic Condition')
        file.write('\n')

        # Add the Sample/Split number, Increment time period, and start & stop VTTI timestamp values
        for i in range(len(split_point_collections)):
            vtti_time_start, vtti_time_stop = split_point_collections[i].start_stop_timestamp()
            file.write('{},{},{},{}'.format(i+1, split_length/60, vtti_time_start, vtti_time_stop))
            file.write('\n')

        file.write('\n')
        file.write(',,,')
        file.write('\n')
        file.write(',,,')
        file.write('\n')

        file.write('Thank you for reviewing the video. Please leave any comments for the video below')
        file.write('\n')
        file.write('LEAVE COMMENTS HERE')

        file.close()

    def export_timeseries_file(self,file):
        """
        Generate time series file of an NDS Point Collection
        :param file: File ready for writing to save time series data
        :return: N/A
        """
        file.write("timestamp,dV [m/s],dX [m],v_lead [m/s],v_foll [m/s],a_lead [m/s2],a_foll [m/s2],headway [s],ttc [s],")
        file.write("latitude,longitude,z_gyro,lane_dist_off_center,wiper,cf_status")

        file.write("\n")

        ttc_list = self.time_to_collision()
        # Write Data to File
        for i in range(self.index):
            # General Car-Following Data
            file.write("{},{},{},{},{},{},{},{},{},".format(self.timestamp[i],
                                                            self.dV[i],
                                                            self.dX[i],
                                                            self.v_lead[i],
                                                            self.v_foll[i],
                                                            self.a_lead[i],
                                                            self.a_foll[i],
                                                            self.headway[i],
                                                            ttc_list[i]))
            # NDS Specific Data
            file.write("{},{},{},{},{},{}".format(self.list_of_data_points[i].vtti_latitude,
                                                  self.list_of_data_points[i].vtti_longitude,
                                                  self.list_of_data_points[i].vtti_gyro_z,
                                                  self.list_of_data_points[i].vtti_lane_distance_off_center,
                                                  self.list_of_data_points[i].vtti_wiper,
                                                  self.list_of_data_points[i].car_following_status))
            file.write("\n")
        file.close()

    def timeseries_summary_plots(self,trip_no):

        # List of measurements for statistics:
        measure_names = ['v_foll', 'a_foll', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway','vtti_time_stamp']
        measure_values = []

        for i in range(len(measure_names)):
            temp = list()
            for j in range(self.point_count()):
                temp.append(getattr(self.list_of_data_points[j],measure_names[i]))
            measure_values.append(temp)
            del temp

        # Plotting
        fig = plt.figure(figsize=(16, 12))  # Size of figure
        fig.suptitle('Time Series Data for Trip {}: {} sec'.format(trip_no,self.time_elapsed()), fontsize=18, fontweight='bold')

        # Network Speed
        speed = plt.subplot2grid((3, 6), (0, 0), colspan=6)
        speed_values = list()
        for i in range(len(measure_values[0])):
            speed_values.append(measure_values[0][i]*3600/1000)  # Convert to km/hr
        speed.plot(measure_values[6],speed_values)
        speed.set_xlabel('Timestamp')
        speed.set_ylabel('Velocity [km/hr]')
        speed.set_title('Speed')
        # Acceleration
        accel = plt.subplot2grid((3, 6), (1, 0), colspan=3)
        accel.plot(measure_values[6],measure_values[1])
        accel.set_xlabel('Timestamp')
        accel.set_ylabel('Acceleration/Deceleration [m/s^2]')
        accel.set_title('Acceleration and Deceleration')
        # Yaw Rate
        yaw = plt.subplot2grid((3, 6), (1, 3), colspan=3)
        yaw.plot(measure_values[6],measure_values[2])
        plt.xlabel('Timestamp')
        plt.ylabel('Yaw Rate/ Gyro [deg/sec]')
        plt.title('Yaw Rate')
        # Lane Offset
        offset = plt.subplot2grid((3, 6), (2, 0), colspan=2)
        offset.plot(measure_values[6],measure_values[3])
        offset.set_xlabel('Timestamp')
        offset.set_ylabel('Lane Offset [cm]')
        offset.set_title('Lane Offset')
        # Wiper
        wiper = plt.subplot2grid((3, 6), (2, 2), colspan=2)
        wiper.plot(measure_values[6],measure_values[4])
        wiper.set_xlabel('Timestamp')
        wiper.set_ylabel('Wiper Status')
        wiper.set_title('Wiper Status')
        # Headway
        hdwy = plt.subplot2grid((3, 6), (2, 4), colspan=2)
        hdwy.plot(measure_values[6],measure_values[5])
        hdwy.set_xlabel('Timestamp')
        hdwy.set_ylabel('Headway [s]')
        hdwy.set_title('Headway ')

        fig.subplots_adjust(wspace=0.7, hspace=0.32)
        plt.close()
        return fig

    ### CF Event Extraction
    def car_following_event_extraction(self, min_cf_time=20, max_cf_dist=60, min_speed=1):
        """
        Function used to automatically extract car-following events from continuous timeseries data.
        :param self: Input data - form of a Collection Object
        :param min_cf_time: minimum car-following collection time [seconds]
        :param max_cf_dist: maximum car-following collection distance [meters]
        :param min_speed: minimum subject vehicle speed [m/s]
        :return: List of Car-Following Event PointCollections
        """
        # Create initial Arrays for aggregating CF Instances
        lead_targets = self.list_of_lead_targets()
        list_of_collections = [WyNdsPointCollection() for row in range(len(lead_targets))]

        # Sort Data Points in CF behavior based on target ID into DataCollections
        for i in range(self.point_count()):
            for j in range(len(lead_targets)):
                if self[i].lead_target_id == lead_targets[j]:
                    try:
                        if self[i].vtti_time_stamp < list_of_collections[j][
                                    list_of_collections[j].point_count() - 1].vtti_time_stamp + 3000:
                            list_of_collections[j].point_append(self[i])
                    except IndexError:
                        list_of_collections[j].point_append(self[i])

        # Checking distance requirements
        temp = list()  # For collecting single instances
        temp_list = list()  # For collecting all instances
        for i in range(len(list_of_collections)):
            for j in range(list_of_collections[i].point_count()):
                if list_of_collections[i][j].dX < max_cf_dist:
                    temp.append(list_of_collections[i][j])
                else:
                    if len(temp) > 0:
                        temp_list.append(WyNdsPointCollection(temp))
                        temp = list()
            if len(temp) > 0:
                temp_list.append(WyNdsPointCollection(temp))
                temp = list()

        list_of_collections = deepcopy(temp_list)

        # Check the subject vehicle's velocity requirements - for freeway driving
        temp = list()  # For collecting single instances
        temp_list = list()  # For collecting all instances
        for i in range(len(list_of_collections)):
            for j in range(list_of_collections[i].point_count()):
                if list_of_collections[i][j].v_foll > min_speed:
                    temp.append(list_of_collections[i][j])
                else:
                    if len(temp) > 0:
                        temp_list.append(WyNdsPointCollection(temp))
                        temp = list()
            if len(temp) > 0:
                temp_list.append(WyNdsPointCollection(temp))
                temp = list()

        list_of_collections = deepcopy(temp_list)

        # Checking time requirements
        temp_list = list()
        for i in range(len(list_of_collections)):
            if list_of_collections[i].time_elapsed() >= min_cf_time:
                temp_list.append(list_of_collections[i])
        list_of_collections = temp_list

        return list_of_collections


class WyNdsReducedPointCollection(PointCollection):

    def car_following_event_extraction(self,min_cf_time=20,max_cf_dist=60,min_speed=1):

        list_of_collections = list()
        temp_collection = WyNdsReducedPointCollection()
        for i in range(len(self.list_of_data_points)):
            if self.list_of_data_points[i].cf_status == 'True':
                temp_collection.point_append(self.list_of_data_points[i])
            else:
                if temp_collection.point_count()>0:
                    list_of_collections.append(temp_collection)
                    temp_collection = WyNdsReducedPointCollection()
        if temp_collection.point_count()>0:
            list_of_collections.append(temp_collection)
        del temp_collection

        # Checking distance requirements
        temp = list()  # For collecting single instances
        temp_list = list()  # For collecting all instances
        for i in range(len(list_of_collections)):
            for j in range(list_of_collections[i].point_count()):
                if list_of_collections[i][j].dX < max_cf_dist:
                    temp.append(list_of_collections[i][j])
                else:
                    if len(temp) > 0:
                        temp_list.append(WyNdsReducedPointCollection(temp))
                        temp = list()
            if len(temp) > 0:
                temp_list.append(WyNdsReducedPointCollection(temp))
                temp = list()

        list_of_collections = deepcopy(temp_list)

        # Check the subject vehicle's velocity requirements - for freeway driving
        temp = list()  # For collecting single instances
        temp_list = list()  # For collecting all instances
        for i in range(len(list_of_collections)):
            for j in range(list_of_collections[i].point_count()):
                if list_of_collections[i][j].v_foll > min_speed:
                    temp.append(list_of_collections[i][j])
                else:
                    if len(temp) > 0:
                        temp_list.append(WyNdsReducedPointCollection(temp))
                        temp = list()
            if len(temp) > 0:
                temp_list.append(WyNdsReducedPointCollection(temp))
                temp = list()

        list_of_collections = deepcopy(temp_list)

        # Checking time requirements
        temp_list = list()
        for i in range(len(list_of_collections)):
            if list_of_collections[i].time_elapsed() >= min_cf_time:
                temp_list.append(list_of_collections[i])
        list_of_collections = temp_list

        return list_of_collections


class WzPointCollection(PointCollection):

    ### CF Event Extraction
    def car_following_event_extraction(self):
        """
        Function used to automatically extract car-following events from WZ Project files, which
        are organized such that all identified car-following events are saved in the file and can be distinguished
        based on the identifying numbers.
        :param self: Input data - form of a Collection Object
        :param min_cf_time: minimum car-following collection time [seconds]
        :param max_cf_dist: maximum car-following collection distance [meters]
        :param min_speed: minimum subject vehicle speed [m/s]
        :return: List of Car-Following Event PointCollections
        """

        identifier = [self.list_of_data_points[0].driver_id,self.list_of_data_points[0].unique_inst,
                          self.list_of_data_points[0].instance_id]
        temp_point_collection = WzPointCollection()
        list_of_collections = list()
        cf_event_counter = 0
        termination_condition = False
        index = 0
        while termination_condition is False:
            try:
                identifier_this = [self.list_of_data_points[index].driver_id,self.list_of_data_points[index].unique_inst,
                              self.list_of_data_points[index].instance_id]
                if identifier == identifier_this:
                    temp_point_collection.point_append([self.list_of_data_points[index]])
                    index += 1
                else:
                    list_of_collections.append(temp_point_collection)
                    del temp_point_collection
                    temp_point_collection = WzPointCollection()
                    identifier = deepcopy(identifier_this)
                    cf_event_counter += 1
            except IndexError:
                termination_condition = True
                list_of_collections.append(temp_point_collection)

        return list_of_collections


class VissimSyntheticCollection(PointCollection):


    def start_stop_timestamp(self):
        start = self.timestamp[0]
        stop = self.timestamp[-1]
        return [start,stop]

    ### CF Event Extraction
    def car_following_event_extraction(self, min_cf_time=20, max_cf_dist=60, min_speed=1):
        """

        :param min_cf_time:
        :param max_cf_dist:
        :param min_speed:
        :return:
        """

        # Create initial list of collections using the v_lead and dX value thresholds
        list_of_collections = list()
        counter = 1
        collection_counter = 0
        while counter < self.index-1:
            single_event = True
            list_of_collections.append(VissimSyntheticCollection())

            while single_event is True:
                if counter == 2475:
                    pass
                if abs(self[counter].v_lead - self[counter-1].v_lead) < 0.7:  # Is CF Event  [m/s2]
                    if abs(self[counter].dX - self[counter-1].dX) < 9:  # Is CF Event  [meters]
                        list_of_collections[collection_counter].point_append(self[counter])
                    else:
                        single_event = False
                else:
                    single_event = False
                if counter < self.index-1:
                    counter += 1
                else:
                    break
            collection_counter += 1

        # Checking distance requirements
        temp = list()  # For collecting single instances
        temp_list = list()  # For collecting all instances
        for i in range(len(list_of_collections)):
            for j in range(list_of_collections[i].point_count()):
                if list_of_collections[i][j].dX < max_cf_dist:
                    temp.append(list_of_collections[i][j])
                else:
                    if len(temp) > 0:
                        temp_list.append(WyNdsPointCollection(temp))
                        temp = list()
            if len(temp) > 0:
                temp_list.append(WyNdsPointCollection(temp))
                temp = list()

        list_of_collections = deepcopy(temp_list)

        # Check the subject vehicle's velocity requirements - for freeway driving
        temp = list()  # For collecting single instances
        temp_list = list()  # For collecting all instances
        for i in range(len(list_of_collections)):
            for j in range(list_of_collections[i].point_count()):
                if list_of_collections[i][j].v_foll > min_speed:
                    temp.append(list_of_collections[i][j])
                else:
                    if len(temp) > 0:
                        temp_list.append(WyNdsPointCollection(temp))
                        temp = list()
            if len(temp) > 0:
                temp_list.append(WyNdsPointCollection(temp))
                temp = list()

        list_of_collections = deepcopy(temp_list)

        # Checking time requirements
        temp_list = list()
        for i in range(len(list_of_collections)):
            if list_of_collections[i].time_elapsed() >= min_cf_time:
                temp_list.append(list_of_collections[i])
        list_of_collections = temp_list

        return list_of_collections


class NewList:
    def __init__(self,new_list):
        self.new_list = new_list
        self.index = len(new_list)

    # Iterator
    def __iter__(self):
        return self

    # Next
    def next(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.list_of_data_points[self.index]

    # Getitem
    def __getitem__(self, index):
            return self.list_of_data_points[index]

    def mean(self):
        return np.nanmean(self.new_list)

    def maximum(self):
        return np.nanmax(self.new_list)

    def minimum(self):
        return np.nanmin(self.new_list)

    def median(self):
        return np.nanmedian(self.new_list)

    def percentile(self,percent=85):
        return np.nanpercentile(self.new_list,percent)

    def standard_deviation(self):
        return np.nanstd(self.new_list)

    def variance(self):
        return self.standard_deviation()**2

    def coeff_variation(self):
        return self.standard_deviation()/self.mean()