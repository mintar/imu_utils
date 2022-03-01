#include "fitallan_acc.h"

using namespace imu;

FitAllanAcc::FitAllanAcc( std::vector< double > sigma2s, std::vector< double > taus, double freq )
: C_Q_(0.0 )
, C_N_(0.0 )
, C_B_(0.0 )
, C_K_(0.0 )
, C_R_(0.0 )
, freq_(freq )
{
    if ( sigma2s.size( ) != taus.size( ) )
        std::cerr << "Error of point size" << std::endl;

    std::vector< double > sigma2s_tmp = checkData( sigma2s, taus );

    std::vector< double > init = initValue( sigma2s_tmp, m_taus );

    int num_samples = sigma2s_tmp.size( );
    double param[] = { init[0], init[1], init[2], init[3], init[4] };

    ceres::Problem problem;

    for ( int i = 0; i < num_samples; ++i )
    {

        //        std::cout << "sigma " << i << " " << taus[i] << " " << sigma2s[i] <<
        //        std::endl;

        ceres::CostFunction* f = new ceres::AutoDiffCostFunction< AllanSigmaError, 1, 5 >(
        new AllanSigmaError( sigma2s_tmp[i], m_taus[i] ) );

        problem.AddResidualBlock( f, NULL /* squared loss */, param );
    }
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.logging_type                 = ceres::SILENT;
    options.trust_region_strategy_type   = ceres::DOGLEG;
    options.max_num_iterations           = 1000;

    ceres::Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );
    //        std::cout << summary.FullReport( ) << "\n";
    //    std::cout << "num_parameters " << summary.num_parameters << std::endl;

    C_Q_ = param[0];
    C_N_ = param[1];
    C_B_ = param[2];
    C_K_ = param[3];
    C_R_ = param[4];

    // std::cout << "C_Q_ " << C_Q_ //
    //           << " " << C_N_  //
    //           << " " << C_B_  //
    //           << " " << C_K_  //
    //           << " " << C_R_ << std::endl;

    std::cout << "================================================================== " << std::endl;
    std::cout << "### Continuous-time Allan variance coefficients" << std::endl;
    std::cout << "Quantization Noise (Q): " << getQ( ) << R"( m / s)" << std::endl;
    std::cout << "White Veloc. Noise (N): " << getN( ) << R"( m / s / sqrt(s)        # Kalibr: \sigma_a, Accelerometer "white noise", accelerometer_noise_density)" << std::endl;
    std::cout << "Bias Instability   (B): " << getB( ) << R"( m / s^2)" << std::endl;
    //std::cout << "Bias Instability   (B): " << getBiasInstability( ) << R"( m / s^2, at )" << taus[findMinIndex( calcSimDeviation( taus ) )] << " s" << std::endl;
    std::cout << "Accel. Random Walk (K): " << getK( ) << R"( m / (s^2 * sqrt(s))  # Kalibr: \sigma_{ba}, Accelerometer "random walk", accelerometer_random_walk)" << std::endl;
    std::cout << "Acceleration Ramp  (R): " << getR( ) << R"( m / s^3)" << std::endl;
    std::cout << "================================================================== " << std::endl;
}

std::vector< double >
FitAllanAcc::initValue( std::vector< double > sigma2s, std::vector< double > taus )
{
    if ( sigma2s.size( ) != taus.size( ) )
        std::cout << " Error with data size!!! " << sigma2s.size( ) << " " << taus.size( ) << std::endl;

    Eigen::MatrixXd Y( sigma2s.size( ), 1 );

    for ( unsigned int index = 0; index < sigma2s.size( ); ++index )
    {
        Y( index, 0 ) = sqrt( sigma2s[index] );
    }
    //        std::cout << "Y " << Y << std::endl;

    int m_order = 2;

    Eigen::MatrixXd B( 2 * m_order + 1, 1 );
    B.setZero( );

    Eigen::MatrixXd F( taus.size( ), 2 * m_order + 1 );
    F.setZero( );

    for ( unsigned int index = 0; index < taus.size( ); ++index )
        for ( int order_index = 0; order_index < 2 * m_order + 1; ++order_index )
        {
            int kk = order_index - m_order;
            F( index, order_index ) = pow( sqrt( taus[index] ), kk );
        }
    //        std::cout << "F " << F << std::endl;

    Eigen::MatrixXd A = F.transpose( ) * F;
    B                 = F.transpose( ) * Y;
    //        std::cout << "B " << B << std::endl;
    //        std::cout << "A " << A << std::endl;

    Eigen::MatrixXd C = A.inverse( ) * B;
    // std::cout << "C " << C.transpose( ) << std::endl;

    std::vector< double > init;
    for ( int index = 0; index < 2 * m_order + 1; ++index )
        init.push_back( std::abs( C( index, 0 ) ) );

    return init;
}

std::vector< double >
FitAllanAcc::calcSimDeviation( const std::vector< double > taus ) const
{
    std::vector< double > des;
    for ( auto& tau : taus )
        des.push_back( sqrt( calcSigma2(C_Q_, C_N_, C_B_, C_K_, C_R_, tau ) ) );
    return des;
}

double
FitAllanAcc::getBiasInstability( ) const
{
    return findMinNum( calcSimDeviation( m_taus ) );
}

std::vector< double >
FitAllanAcc::checkData( std::vector< double > sigma2s, std::vector< double > taus )
{
    std::vector< double > sigma2s_tmp;
    double data_tmp = 0;
    //        bool is_first   = true;
    for ( unsigned int index = 0; index < sigma2s.size( ); ++index )
    {
        if ( taus[index] < 1 )
        {
            if ( data_tmp < sigma2s[index] )
            {
                data_tmp = sigma2s[index];
                continue;
            }
            else
            {
                sigma2s_tmp.push_back( sigma2s[index] );
                m_taus.push_back( taus[index] );
            }
        }
        else
        {
            sigma2s_tmp.push_back( sigma2s[index] );
            m_taus.push_back( taus[index] );
        }
    }
    return sigma2s_tmp;
}

double
FitAllanAcc::findMinNum( const std::vector< double > num ) const
{
    double min = 1000.0;
    for ( unsigned int index = 0; index < num.size( ); ++index )
        min = min < num[index] ? min : num[index];
    return min;
}

int
FitAllanAcc::findMinIndex( std::vector< double > num )
{
    double min    = 1000.0;
    int min_index = 0;
    for ( unsigned int index = 0; index < num.size( ); ++index )
    {
        min_index = min < num[index] ? min_index : index;
        min       = min < num[index] ? min : num[index];
    }
    return min_index;
}

double
FitAllanAcc::calcSigma2(double C_Q, double C_N, double C_B, double C_K, double C_R, double tau ) const
{
    // clang-format off
  return  C_Q * C_Q / ( tau * tau )
      + C_N * C_N / tau
      + C_B * C_B
      + C_K * C_K * tau
      + C_R * C_R * tau * tau;
    // clang-format on
}

double
FitAllanAcc::getQ( ) const
{
    return sqrt(C_Q_ * C_Q_ ) / ( sqrt(3.0 ) );
}

double
FitAllanAcc::getN( ) const
{
    return sqrt(C_N_ * C_N_ );
}

double
FitAllanAcc::getB( ) const
{
    return sqrt(C_B_ * C_B_ * M_PI / (2 * log(2)));
}

double
FitAllanAcc::getK( ) const
{
    return sqrt( 3.0 * C_K_ * C_K_ );
}

double
FitAllanAcc::getR( ) const
{
    return sqrt( 2.0 * C_R_ * C_R_ );
}
