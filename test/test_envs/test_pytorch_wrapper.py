import unittest
from trading_gym.utils.data.toy import create_toy_data
from trading_gym.envs.portfolio_gym.portfolio_gym import PortfolioTradingGym

order_book_id_number = 100
toy_data = create_toy_data(order_book_ids_number=order_book_id_number, feature_number=10, start="2019-05-01", end="2019-12-12", frequency="D")


class TestDataWraper(unittest.TestCase):
    
    def setUp(self):
        self.order_book_id_number = 100
        self.feature_number = 10
        self.toy_data = create_toy_data(order_book_ids_number=order_book_id_number, feature_number=self.feature_number, start="2019-05-01", end="2019-12-12", frequency="D")
        self.env_2d = PortfolioTradingGym(data_df=self.toy_data, sequence_window=1, add_cash=False, mode="numpy")
        
        self.sequence_window =3
        self.env_3d = PortfolioTradingGym(data_df = self.toy_data, sequence_window=self.sequence_window, add_cash=False, mode="numpy")
    
    def test_sequence_window_equal_1(self):
        state = self.env_2d.reset()
        self.assertEqual(state.shape,(self.order_book_id_number, self.feature_number))
        
        random_action = self.env_2d.action_space.sample()
        next_state, reward, done, info = self.env_2d.step(random_action)
        
        self.assertEqual(next_state.shape, (self.order_book_id_number, self.feature_number))
        return next_state
    
    def test_sequence_window_mt_1(self):
        state = self.env_3d.reset()
        self.assertEqual(state.shape,(self.order_book_id_number, self.sequence_window, self.feature_number))
        
        random_action = self.env_3d.action_space.sample()
        next_state, reward, done, info = self.env_3d.step(random_action)
        
        self.assertEqual(next_state.shape, (self.order_book_id_number, self.sequence_window, self.feature_number))
    
    def test_torch_wrapper(self):
        
    
if __name__ == "__main__":
    
    tester = TestDataWraper()
    tester.setUp()
    state = tester.test_sequence_equal_1()


