**Project in progress**

# RNN_LSTM_KDD99
This project aims to reproduce the results made by RC Staudemeyer in his article "Applying machine learning principles to the information security field through intelligent intrusion detection systems.".


## Project structure
### Required packages:
- tensorflow (version used: 1.13.1)
- sklearn (version used: 0.0)  
- numpy (version used: 1.16.2) 
- pandas (version used: 0.24.2)
- Jupyter (version used: 1.0.0)
- configparser (version used: 3.7.4)
.... [To be completed]

### Directory structure:
[To be completed]

## Data
The following freely available dataset will be used in its corrected version : [KDD Cup](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) (1999).

### Inputs

The KDD's data which includes 40 values, 32 continuous, 8 of which are symbolic and will have to binarized if they aren't already. While some value are described as continuous in the dataset, it is important to note that it isn't the mathematical definition in the sense that { 0, 1, 2, 3, 4 } will be described as continuous.

When symbolic variables are binarized, the resulting vector contains 3 (protocol) + 70 (service) + 11 (flag) + 37 (continuous or already binarized) = 121 values

| Variable name               | Discrete or Continuous | Possible values  |
| --------------------------- |:----------------------:| ----------------:|
| Duration                    | Continuous             | [0, ∞[           |
| Protocol Type               | Discrete (Symbolic)    | {icmp, tcp, udp} |
| Service                     | Discrete (Symbolic)    | {IRC, X11, Z39_50, aol, ..., hostnames, http, ftp}      |
| Flag                        | Discrete (Symbolic)    | {OTH, REJ, RSTO, RSTOS0, RSTR, S0, S1, S2, S3, SF, SH}  |
| Source bytes                | Continuous             | [0, ∞[           |
| Destination bytes           | Continuous             | [0, ∞[           |
| Land                        | Discrete               | {0, 1} |
| Wrong fragment              | Continuous             | [0, 3] |
| Urgent                      | Continuous             | [0, 5] |
| Hot                         | Continuous             | [0, 77] |
| Num failed                  | Continuous             | [0, 5] |
| Logged in                   | Discrete               | {0, 1} |
| Num compromised             | Continuous             | [0, 1] |
| Root shell                  | Continuous             | [0, ∞[ |
| Su attempted                | Discrete               | {0, 1} |
| Num root                    | Continuous             | [0, 2] |
| Num file creations          | Continuous             | [0, 40] |
| Num shells                  | Continuous             | [0, 2] |
| Num access files            | Continuous             | [0, 9] |
| Num outbound cmds           | Continuous             | { 0 } |
| Is host login               | Discrete               | {0, 1} |
| Is guest login              | Discrete               | {0, 1} |
| Count                       | Countinuous            | [0, 511] |
| Srv count                   | Countinuous            | [0, 511] |
| Serror rate                 | Countinuous            | [0, 1] |
| Srv serror rate             | Countinuous            | [0, 1] |
| Rerror rate                 | Countinuous            | [0, 1] |
| Srv rerror rate             | Countinuous            | [0, 1] |
| Same srv rate               | Countinuous            | [0, 1] |
| Diff srv rate               | Countinuous            | [0, 1] |
| Srv diff host rate          | Countinuous            | [0, 1] |
| Dst host count              | Countinuous            | [0, 255] |
| Dst host srv count          | Countinuous            | [0, 255] |
| Dst host same srv rate      | Countinuous            | [0, 1] |
| Dst host diff srv rate      | Countinuous            | [0, 1] |
| Dst host same src port rate | Countinuous            | [0, 1] |
| Dst host serror rate        | Countinuous            | [0, 1] |
| Dst host srv serror rate    | Countinuous            | [0, 1] |
| Dst host rerror rate        | Countinuous            | [0, 1] |
| Dst host srv rerror rate    | Countinuous            | [0, 1] |

### Outputs

There are multiple possible outputs that can be grouped as follows:

| Label       | Description                         | Sub-labels |
| ----------- |:-----------------------------------:| ---------- |
| normal      | Normal                              | { normal } |
| probe       | Probe                               | { ipsweep, nmap, postsweep, satan, saint, mscan } |
| dos         | Denial of Service Attack            | { back, land, neptune, pod, smurf, teardrop, apache2, udpstorm, processtable, mailbomb } |
| u2r         | User to root (Privilege escalation) | { buffer_overflow, loadmodule, perl, rootkit, xterm, ps, sqlattack } |
| r2l         | Remote to user                      | { ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster, snmpgetattack, named, xlock, xsnoop, sendmail, httptunnel, worm, snmpguess } |

### Considerations over the KDD Cup 99 dataset

While one of the few datasets freely available on the subject, the KDD Cup 99 dataset has many shortcomings that can be summarized to:

- Categories are unbalanced, 70% of the testing set is DoS
- Detecting DoS is overall pointless as it generates a log of traffic
- Redundant records

All those points and many more are described in "A detailed analysis of the KDD Cup 99 Data Set" by Mahbod Tavallaee, Ebrahim Bagheri, Wei Lu, and Ali A. Ghorbani.
They proposed a solution which will be partially used namely:

- Removing redundant records
- Artificially increase the number of R2L and U2R examples

(Considerations inspired by [Edouard Belval](https://github.com/Belval/ML-IDS))

## References
This projet has been inspired by the work "Applying long short-term memory recurrent neural networks to intrusion detection" by RC Staudemeyer - ‎2015 as well as both the repositories from [Jiachuan Deng](https://github.com/JiachuanDENG/KDDCup99_NID_LSTM) and [Edouard Belval](https://github.com/Belval/ML-IDS).

## License
[MIT](LICENSE) © [Sylvain Lapeyrade](https://github.com/sylvainlapeyrade)
