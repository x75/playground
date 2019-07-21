#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
GPX TRACK VISUALIZER
ideagora geomatics-2018
http://www.geodose.com/
""" 

import time, sys, math

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from xml.dom import minidom

# from IPython import display

if len(sys.argv) > 1:
    gpxfile = sys.argv[1]
else:
    gpxfile = '/home/x75/schwanderberg-gigerhorn-1.gpx'

# READ GPX FILE
data=open(gpxfile, 'r') #CHANGE YOUR FILE HERE
xmldoc = minidom.parse(data)
track = xmldoc.getElementsByTagName('trkpt')
elevation=xmldoc.getElementsByTagName('ele')
datetime=xmldoc.getElementsByTagName('time')
n_track=len(track)

#PARSING GPX ELEMENT
lon_list=[]
lat_list=[]
h_list=[]
time_list=[]
for s in range(n_track):
    lon,lat=track[s].attributes['lon'].value,track[s].attributes['lat'].value
    elev=elevation[s].firstChild.nodeValue
    lon_list.append(float(lon))
    lat_list.append(float(lat))
    h_list.append(float(elev))
    # PARSING TIME ELEMENT
    dt=datetime[s].firstChild.nodeValue
    time_split=dt.split('T')
    hms_split=time_split[1].split(':')
    time_hour=int(hms_split[0])
    time_minute=int(hms_split[1])
    time_second=int(hms_split[2].split('Z')[0])
    total_second=time_hour*3600+time_minute*60+time_second
    time_list.append(total_second)
    

#GEODETIC TO CARTERSIAN FUNCTION
def geo2cart(lon,lat,h):
    a=6378137 #WGS 84 Major axis
    b=6356752.3142 #WGS 84 Minor axis
    e2=1-(b**2/a**2)
    N=float(a/math.sqrt(1-e2*(math.sin(math.radians(abs(lat)))**2)))
    X=(N+h)*math.cos(math.radians(lat))*math.cos(math.radians(lon))
    Y=(N+h)*math.cos(math.radians(lat))*math.sin(math.radians(lon))
    return X,Y

#DISTANCE FUNCTION
def distance(x1,y1,x2,y2):
    d=math.sqrt((x1-x2)**2+(y1-y2)**2)
    return d

#SPEED FUNCTION
def speed(x0,y0,x1,y1,t0,t1):
    d=math.sqrt((x0-x1)**2+(y0-y1)**2)
    delta_t=t1-t0
    s=float(d/delta_t)
    return s

#POPULATE DISTANCE AND SPEED LIST
d_list=[0.0]
speed_list=[0.0]
l=0
for k in range(n_track-1):
    if k<(n_track-1):
        l=k+1
    else:
        l=k
    XY0=geo2cart(lon_list[k],lat_list[k],h_list[k])
    XY1=geo2cart(lon_list[l],lat_list[l],h_list[l])
    
    #DISTANCE
    d=distance(XY0[0],XY0[1],XY1[0],XY1[1])
    sum_d=d+d_list[-1]
    d_list.append(sum_d)
    
    #SPEED
    s=speed(XY0[0],XY0[1],XY1[0],XY1[1],time_list[k],time_list[l])
    speed_list.append(s)

#PLOT TRACK
# plt.ion()
# f,(track,speed,elevation)=plt.subplots(3,1)



# fig = plt.figure()
# ax = fig.add_subplot(111)

# # some X and Y data
# x = [0]
# y = [0]

# li, = ax.plot(x, y,'o')

# # draw and show it
# fig.canvas.draw()
# plt.show(block=False)


# # loop to update the data
# for i in range(100):
#     try:
#         x.append(i)
#         y.append(i)

#         # set the new data
#         li.set_xdata(x)
#         li.set_ydata(y)

#         ax.relim() 
#         ax.autoscale_view(True,True,True) 

#         fig.canvas.draw()

#         time.sleep(0.01)
#     except KeyboardInterrupt:
#         plt.close('all')
#         break


# plt.ion()

fig = plt.figure()
# plt.show()
fig.set_figheight(12)
fig.set_figwidth(16)


gridspec = GridSpec(2, 2)
# subplot(subplotspec)

# ax_track = fig.add_subplot(3,1,1)
# ax_speed = fig.add_subplot(3,1,2)
# ax_elevation = fig.add_subplot(3,1,3)

subplotspec = gridspec.new_subplotspec((0, 0), 2, 1)
ax_track = fig.add_subplot(subplotspec)
subplotspec = gridspec.new_subplotspec((0, 1), 1, 1)
ax_speed = fig.add_subplot(subplotspec)
subplotspec = gridspec.new_subplotspec((1, 1), 1, 1)
ax_elevation = fig.add_subplot(subplotspec)

# fig.show()

# draw and show it
fig.canvas.draw()
plt.show(block=False)

# sys.exit()

# plt.subplots_adjust(hspace=0.5)

ax_track.plot(lon_list,lat_list,'k', linestyle='-', marker=',', alpha=0.75)
ax_track.set_ylabel("Latitude")
ax_track.set_xlabel("Longitude")
ax_track.set_title("Track Plot")
ax_track.set_aspect(1)

#PLOT SPEED
ax_speed.bar(d_list,speed_list,30,color='w',edgecolor='w')
ax_speed.set_title("Speed")
ax_speed.set_xlabel("Distance(m)")
ax_speed.set_ylabel("Speed(m/s)")

#PLOT ELEVATION PROFILE
base_reg=0
ax_elevation.plot(d_list,h_list)
ax_elevation.fill_between(d_list,h_list,base_reg,alpha=0.1)
ax_elevation.set_title("Elevation Profile")
ax_elevation.set_xlabel("Distance(m)")
ax_elevation.set_ylabel("GPS Elevation(m)")
ax_elevation.grid()

# plt.show()
# plt.ion()

# ANIMATION/DYNAMIC PLOT
for i in range(n_track):
    print('lon, lat', lon_list[i],lat_list[i])
    ax_track.plot(lon_list[i],lat_list[i],'yo', alpha=0.5)
    ax_speed.bar(d_list[i],speed_list[i],30,color='g',edgecolor='g')
    ax_elevation.plot(d_list[i],h_list[i],'ro')
    # display.display(plt.gcf())
    # display.clear_output(wait=True)
    fig.canvas.draw()
    # plt.draw()
    time.sleep(0.001)

plt.ioff()
# plt.show()


# In[ ]:




